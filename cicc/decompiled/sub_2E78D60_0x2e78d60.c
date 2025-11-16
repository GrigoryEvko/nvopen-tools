// Function: sub_2E78D60
// Address: 0x2e78d60
//
void __fastcall sub_2E78D60(__int64 a1)
{
  __int64 v1; // rdi
  void (*v2)(void); // rax

  v1 = *(_QWORD *)(a1 + 672);
  if ( v1 )
  {
    v2 = *(void (**)(void))(*(_QWORD *)v1 + 40LL);
    if ( v2 != nullsub_1598 )
      v2();
  }
}
