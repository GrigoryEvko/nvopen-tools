// Function: sub_287B9A0
// Address: 0x287b9a0
//
__int64 __fastcall sub_287B9A0(pthread_rwlock_t *rwlock)
{
  __int64 v1; // rax
  __int64 v2; // r12

  sub_DFEA20((__int64)rwlock);
  sub_B1A2E0((__int64)rwlock);
  sub_D9A960((__int64)rwlock);
  sub_D4AA90((__int64)rwlock);
  sub_F67EE0((__int64)rwlock);
  v1 = sub_22077B0(0x38u);
  v2 = v1;
  if ( v1 )
  {
    *(_QWORD *)(v1 + 8) = 23;
    *(_QWORD *)v1 = "Loop Terminator Folding";
    *(_QWORD *)(v1 + 16) = "loop-term-fold";
    *(_QWORD *)(v1 + 32) = &unk_500192C;
    *(_WORD *)(v1 + 40) = 0;
    *(_QWORD *)(v1 + 24) = 14;
    *(_QWORD *)(v1 + 48) = sub_287BF10;
  }
  sub_BC3090(rwlock, (_QWORD *)v1, 1);
  return v2;
}
