// Function: sub_DF4D20
// Address: 0xdf4d20
//
__int64 __fastcall sub_DF4D20(pthread_rwlock_t *rwlock)
{
  __int64 v1; // rax
  __int64 v2; // r12

  v1 = sub_22077B0(56);
  v2 = v1;
  if ( v1 )
  {
    *(_QWORD *)(v1 + 8) = 29;
    *(_QWORD *)v1 = "Scoped NoAlias Alias Analysis";
    *(_QWORD *)(v1 + 16) = "scoped-noalias-aa";
    *(_QWORD *)(v1 + 24) = 17;
    *(_QWORD *)(v1 + 32) = &unk_4F89B44;
    *(_WORD *)(v1 + 40) = 256;
    *(_QWORD *)(v1 + 48) = sub_DF5180;
  }
  sub_BC3090(rwlock, (_QWORD *)v1, 1);
  return v2;
}
