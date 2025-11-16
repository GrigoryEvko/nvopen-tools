// Function: sub_2DE6900
// Address: 0x2de6900
//
__int64 __fastcall sub_2DE6900(pthread_rwlock_t *rwlock)
{
  __int64 v1; // rax
  __int64 v2; // r12

  sub_B1A2E0((__int64)rwlock);
  v1 = sub_22077B0(0x38u);
  v2 = v1;
  if ( v1 )
  {
    *(_QWORD *)(v1 + 8) = 63;
    *(_QWORD *)v1 = "Lower interleaved memory accesses to target specific intrinsics";
    *(_QWORD *)(v1 + 16) = "interleaved-access";
    *(_QWORD *)(v1 + 32) = &unk_501E754;
    *(_WORD *)(v1 + 40) = 0;
    *(_QWORD *)(v1 + 24) = 18;
    *(_QWORD *)(v1 + 48) = sub_2DE7160;
  }
  sub_BC3090(rwlock, (_QWORD *)v1, 1);
  return v2;
}
