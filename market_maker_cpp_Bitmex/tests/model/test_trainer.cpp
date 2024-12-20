#include <gtest/gtest.h>
#include <market_maker/model/trainer.h>

class ModelTrainerTest : public ::testing::Test {
protected:
    void SetUp() override {
        args.d_model = 64;
        args.num_epochs = 2;
        args.batch_size = 32;
        args.learning_rate = 0.001;
        
        model = std::make_shared<SSMHippo>(args);
        optimizer = std::make_shared<torch::optim::Adam>(
            model->parameters(), args.learning_rate);
        trainer = std::make_unique<ModelTrainer>(args, model, optimizer);
    }
    
    ModelArgs args;
    std::shared_ptr<SSMHippo> model;
    std::shared_ptr<torch::optim::Optimizer> optimizer;
    std::unique_ptr<ModelTrainer> trainer;
};

TEST_F(ModelTrainerTest, Training) {
    torch::Tensor train_data = torch::randn({100, 4, 128});
    torch::Tensor val_data = torch::randn({20, 4, 128});
    
    float initial_loss = trainer->validate(val_data);
    trainer->train(train_data, val_data, "test_checkpoint.pt");
    float final_loss = trainer->validate(val_data);
    
    EXPECT_LT(final_loss, initial_loss);
}

TEST_F(ModelTrainerTest, Checkpointing) {
    torch::Tensor data = torch::randn({100, 4, 128});
    
    trainer->train_epoch(data);
    trainer->save_checkpoint("test_checkpoint.pt");
    
    auto new_trainer = std::make_unique<ModelTrainer>(args, model, optimizer);
    new_trainer->load_checkpoint("test_checkpoint.pt");
    
    EXPECT_EQ(trainer->get_current_epoch(), new_trainer->get_current_epoch());
}

TEST_F(ModelTrainerTest, LearningRateScheduling) {
    torch::Tensor data = torch::randn({100, 4, 128});
    
    float initial_lr = args.learning_rate;
    trainer->train_epoch(data);
    trainer->update_learning_rate();
    
    float current_lr = optimizer->param_groups()[0].options().get_lr();
    EXPECT_NE(current_lr, initial_lr);
} 