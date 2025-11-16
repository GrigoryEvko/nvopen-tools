// Function: sub_28E4160
// Address: 0x28e4160
//
__int64 __fastcall sub_28E4160(pthread_rwlock_t *rwlock)
{
  __int64 v1; // rax
  __int64 v2; // r12

  sub_97FFF0((__int64)rwlock);
  sub_B1A2E0((__int64)rwlock);
  sub_DFEA20((__int64)rwlock);
  sub_1049990((__int64)rwlock);
  v1 = sub_22077B0(0x38u);
  v2 = v1;
  if ( v1 )
  {
    *(_QWORD *)(v1 + 8) = 43;
    *(_QWORD *)v1 = "Partially inline calls to library functions";
    *(_QWORD *)(v1 + 16) = "partially-inline-libcalls";
    *(_QWORD *)(v1 + 32) = &unk_5004670;
    *(_WORD *)(v1 + 40) = 0;
    *(_QWORD *)(v1 + 24) = 25;
    *(_QWORD *)(v1 + 48) = sub_28E4510;
  }
  sub_BC3090(rwlock, (_QWORD *)v1, 1);
  return v2;
}
