// Function: init_routine
// Address: 0x223dd50
//
void init_routine(void)
{
  (*(void (**)(void))(__readfsqword(0) - 32))();
}
