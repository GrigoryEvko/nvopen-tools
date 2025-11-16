// Function: sub_2778270
// Address: 0x2778270
//
__int64 __fastcall sub_2778270(pthread_rwlock_t *rwlock)
{
  __int64 v1; // rax
  __int64 v2; // r12

  sub_DFEA20((__int64)rwlock);
  sub_CFB980((__int64)rwlock);
  sub_B1A2E0((__int64)rwlock);
  sub_97FFF0((__int64)rwlock);
  v1 = sub_22077B0(0x38u);
  v2 = v1;
  if ( v1 )
  {
    *(_QWORD *)(v1 + 8) = 9;
    *(_QWORD *)v1 = "Early CSE";
    *(_QWORD *)(v1 + 16) = "early-cse";
    *(_QWORD *)(v1 + 32) = &unk_4FFB0F4;
    *(_WORD *)(v1 + 40) = 0;
    *(_QWORD *)(v1 + 24) = 9;
    *(_QWORD *)(v1 + 48) = sub_277B910;
  }
  sub_BC3090(rwlock, (_QWORD *)v1, 1);
  return v2;
}
