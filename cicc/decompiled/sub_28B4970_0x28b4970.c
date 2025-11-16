// Function: sub_28B4970
// Address: 0x28b4970
//
__int64 __fastcall sub_28B4970(pthread_rwlock_t *rwlock)
{
  __int64 v1; // rax
  __int64 v2; // r12

  sub_97FFF0((__int64)rwlock);
  sub_DFEA20((__int64)rwlock);
  sub_CF6DB0((__int64)rwlock);
  v1 = sub_22077B0(0x38u);
  v2 = v1;
  if ( v1 )
  {
    *(_QWORD *)(v1 + 8) = 36;
    *(_QWORD *)v1 = "Merge contiguous icmps into a memcmp";
    *(_QWORD *)(v1 + 16) = "mergeicmps";
    *(_QWORD *)(v1 + 32) = &unk_50044AC;
    *(_WORD *)(v1 + 40) = 0;
    *(_QWORD *)(v1 + 24) = 10;
    *(_QWORD *)(v1 + 48) = sub_28BB330;
  }
  sub_BC3090(rwlock, (_QWORD *)v1, 1);
  return v2;
}
