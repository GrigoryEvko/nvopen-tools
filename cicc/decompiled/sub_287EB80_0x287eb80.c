// Function: sub_287EB80
// Address: 0x287eb80
//
__int64 __fastcall sub_287EB80(pthread_rwlock_t *rwlock)
{
  __int64 v1; // rax
  __int64 v2; // r12

  sub_CFB980((__int64)rwlock);
  sub_F6DC10((__int64)rwlock);
  sub_DFEA20((__int64)rwlock);
  v1 = sub_22077B0(0x38u);
  v2 = v1;
  if ( v1 )
  {
    *(_QWORD *)(v1 + 8) = 12;
    *(_QWORD *)v1 = "Unroll loops";
    *(_QWORD *)(v1 + 16) = "loop-unroll";
    *(_QWORD *)(v1 + 32) = &unk_5001CAC;
    *(_WORD *)(v1 + 40) = 0;
    *(_QWORD *)(v1 + 24) = 11;
    *(_QWORD *)(v1 + 48) = sub_2880F30;
  }
  sub_BC3090(rwlock, (_QWORD *)v1, 1);
  return v2;
}
