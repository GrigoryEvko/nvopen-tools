// Function: sub_2F28F60
// Address: 0x2f28f60
//
__int64 __fastcall sub_2F28F60(pthread_rwlock_t *rwlock)
{
  __int64 v1; // rax
  __int64 v2; // r12

  sub_2E6D3E0((__int64)rwlock);
  sub_2EA61A0((__int64)rwlock);
  v1 = sub_22077B0(0x38u);
  v2 = v1;
  if ( v1 )
  {
    *(_QWORD *)(v1 + 8) = 22;
    *(_QWORD *)v1 = "Peephole Optimizations";
    *(_QWORD *)(v1 + 16) = "peephole-opt";
    *(_QWORD *)(v1 + 32) = &unk_50226F4;
    *(_WORD *)(v1 + 40) = 0;
    *(_QWORD *)(v1 + 24) = 12;
    *(_QWORD *)(v1 + 48) = sub_2F2B850;
  }
  sub_BC3090(rwlock, (_QWORD *)v1, 1);
  return v2;
}
