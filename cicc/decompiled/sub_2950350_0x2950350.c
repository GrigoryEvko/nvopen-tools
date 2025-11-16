// Function: sub_2950350
// Address: 0x2950350
//
__int64 __fastcall sub_2950350(pthread_rwlock_t *rwlock)
{
  __int64 v1; // rax
  __int64 v2; // r12

  sub_B1A2E0((__int64)rwlock);
  sub_D9A960((__int64)rwlock);
  sub_DFEA20((__int64)rwlock);
  sub_D4AA90((__int64)rwlock);
  sub_97FFF0((__int64)rwlock);
  v1 = sub_22077B0(0x38u);
  v2 = v1;
  if ( v1 )
  {
    *(_QWORD *)(v1 + 8) = 66;
    *(_QWORD *)v1 = "Split GEPs to a variadic base and a constant offset for better CSE";
    *(_QWORD *)(v1 + 16) = "separate-const-offset-from-gep";
    *(_QWORD *)(v1 + 32) = &unk_5005724;
    *(_WORD *)(v1 + 40) = 0;
    *(_QWORD *)(v1 + 24) = 30;
    *(_QWORD *)(v1 + 48) = sub_2951F20;
  }
  sub_BC3090(rwlock, (_QWORD *)v1, 1);
  return v2;
}
