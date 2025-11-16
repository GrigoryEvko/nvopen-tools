// Function: sub_27EB560
// Address: 0x27eb560
//
__int64 __fastcall sub_27EB560(pthread_rwlock_t *rwlock)
{
  __int64 v1; // rax
  __int64 v2; // r12

  sub_F6DC10((__int64)rwlock);
  sub_97FFF0((__int64)rwlock);
  sub_DFEA20((__int64)rwlock);
  sub_103BB40((__int64)rwlock);
  sub_1027A60((__int64)rwlock);
  v1 = sub_22077B0(0x38u);
  v2 = v1;
  if ( v1 )
  {
    *(_QWORD *)(v1 + 8) = 26;
    *(_QWORD *)v1 = "Loop Invariant Code Motion";
    *(_QWORD *)(v1 + 16) = "licm";
    *(_QWORD *)(v1 + 32) = &unk_4FFDD4C;
    *(_WORD *)(v1 + 40) = 0;
    *(_QWORD *)(v1 + 24) = 4;
    *(_QWORD *)(v1 + 48) = sub_27EDFF0;
  }
  sub_BC3090(rwlock, (_QWORD *)v1, 1);
  return v2;
}
