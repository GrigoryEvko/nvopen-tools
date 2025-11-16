// Function: sub_CF6C70
// Address: 0xcf6c70
//
__int64 __fastcall sub_CF6C70(pthread_rwlock_t *rwlock)
{
  __int64 v1; // rax
  __int64 v2; // r12

  sub_D05480();
  sub_CF69A0((__int64)rwlock);
  sub_D1D8C0(rwlock);
  sub_DF4610(rwlock);
  sub_DF5010(rwlock);
  sub_E00570(rwlock);
  v1 = sub_22077B0(56);
  v2 = v1;
  if ( v1 )
  {
    *(_QWORD *)(v1 + 8) = 31;
    *(_QWORD *)v1 = "Function Alias Analysis Results";
    *(_QWORD *)(v1 + 16) = "aa";
    *(_QWORD *)(v1 + 24) = 2;
    *(_QWORD *)(v1 + 32) = &unk_4F86530;
    *(_WORD *)(v1 + 40) = 256;
    *(_QWORD *)(v1 + 48) = sub_CF6F20;
  }
  sub_BC3090(rwlock, (_QWORD *)v1, 1);
  return v2;
}
