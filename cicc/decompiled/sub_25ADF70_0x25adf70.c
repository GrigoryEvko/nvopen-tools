// Function: sub_25ADF70
// Address: 0x25adf70
//
__int64 __fastcall sub_25ADF70(pthread_rwlock_t *rwlock)
{
  __int64 v1; // rax
  __int64 v2; // r12

  v1 = sub_22077B0(0x38u);
  v2 = v1;
  if ( v1 )
  {
    *(_QWORD *)(v1 + 8) = 25;
    *(_QWORD *)v1 = "Dead Argument Elimination";
    *(_QWORD *)(v1 + 16) = "deadargelim";
    *(_QWORD *)(v1 + 32) = &unk_4FEFCF4;
    *(_WORD *)(v1 + 40) = 0;
    *(_QWORD *)(v1 + 24) = 11;
    *(_QWORD *)(v1 + 48) = sub_25AF740;
  }
  sub_BC3090(rwlock, (_QWORD *)v1, 1);
  return v2;
}
