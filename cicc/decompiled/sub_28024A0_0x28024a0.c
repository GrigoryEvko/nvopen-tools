// Function: sub_28024A0
// Address: 0x28024a0
//
__int64 __fastcall sub_28024A0(pthread_rwlock_t *rwlock)
{
  __int64 v1; // rax
  __int64 v2; // r12

  sub_CFB980((__int64)rwlock);
  sub_DFEA20((__int64)rwlock);
  sub_D4AA90((__int64)rwlock);
  sub_F67EE0((__int64)rwlock);
  sub_1049990((__int64)rwlock);
  sub_D9A960((__int64)rwlock);
  v1 = sub_22077B0(0x38u);
  v2 = v1;
  if ( v1 )
  {
    *(_QWORD *)(v1 + 8) = 18;
    *(_QWORD *)v1 = "Loop Data Prefetch";
    *(_QWORD *)(v1 + 16) = "loop-data-prefetch";
    *(_QWORD *)(v1 + 32) = &unk_4FFE6EC;
    *(_WORD *)(v1 + 40) = 0;
    *(_QWORD *)(v1 + 24) = 18;
    *(_QWORD *)(v1 + 48) = sub_2802A00;
  }
  sub_BC3090(rwlock, (_QWORD *)v1, 1);
  return v2;
}
