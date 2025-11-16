// Function: sub_2E44830
// Address: 0x2e44830
//
__int64 __fastcall sub_2E44830(pthread_rwlock_t *rwlock)
{
  __int64 v1; // rax
  __int64 v2; // r12

  v1 = sub_22077B0(0x38u);
  v2 = v1;
  if ( v1 )
  {
    *(_QWORD *)(v1 + 8) = 29;
    *(_QWORD *)v1 = "Machine Copy Propagation Pass";
    *(_QWORD *)(v1 + 16) = "machine-cp";
    *(_QWORD *)(v1 + 32) = &unk_501F390;
    *(_WORD *)(v1 + 40) = 0;
    *(_QWORD *)(v1 + 24) = 10;
    *(_QWORD *)(v1 + 48) = sub_2E44C90;
  }
  sub_BC3090(rwlock, (_QWORD *)v1, 1);
  return v2;
}
