// Function: sub_EA1B30
// Address: 0xea1b30
//
void __fastcall __noreturn sub_EA1B30(__int64 a1)
{
  void (*v1)(void); // rax

  v1 = *(void (**)(void))(**(_QWORD **)(*(_QWORD *)(a1 + 296) + 8LL) + 208LL);
  if ( v1 != nullsub_330 )
    v1();
  BUG();
}
