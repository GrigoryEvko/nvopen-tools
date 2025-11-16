// Function: sub_2973840
// Address: 0x2973840
//
__int64 __fastcall sub_2973840(pthread_rwlock_t *rwlock)
{
  __int64 v1; // rax
  __int64 v2; // r12

  sub_DFEA20((__int64)rwlock);
  sub_CFB980((__int64)rwlock);
  sub_B1A2E0((__int64)rwlock);
  sub_10564E0((__int64)rwlock);
  v1 = sub_22077B0(0x38u);
  v2 = v1;
  if ( v1 )
  {
    *(_QWORD *)(v1 + 8) = 16;
    *(_QWORD *)v1 = "Simplify the CFG";
    *(_QWORD *)(v1 + 16) = "simplifycfg";
    *(_QWORD *)(v1 + 32) = &unk_500660C;
    *(_WORD *)(v1 + 40) = 0;
    *(_QWORD *)(v1 + 24) = 11;
    *(_QWORD *)(v1 + 48) = sub_2974E40;
  }
  sub_BC3090(rwlock, (_QWORD *)v1, 1);
  return v2;
}
