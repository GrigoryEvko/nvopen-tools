// Function: sub_DF36F0
// Address: 0xdf36f0
//
__int64 __fastcall sub_DF36F0(pthread_rwlock_t *rwlock)
{
  __int64 v1; // rax
  __int64 v2; // r12

  sub_D9A960((__int64)rwlock);
  v1 = sub_22077B0(56);
  v2 = v1;
  if ( v1 )
  {
    *(_QWORD *)(v1 + 8) = 36;
    *(_QWORD *)v1 = "ScalarEvolution-based Alias Analysis";
    *(_QWORD *)(v1 + 16) = "scev-aa";
    *(_QWORD *)(v1 + 24) = 7;
    *(_QWORD *)(v1 + 32) = &unk_4F89B30;
    *(_WORD *)(v1 + 40) = 256;
    *(_QWORD *)(v1 + 48) = sub_DF4780;
  }
  sub_BC3090(rwlock, (_QWORD *)v1, 1);
  return v2;
}
