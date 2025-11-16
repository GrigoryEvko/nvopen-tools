// Function: sub_3573E50
// Address: 0x3573e50
//
__int64 __fastcall sub_3573E50(pthread_rwlock_t *rwlock)
{
  __int64 v1; // rax
  __int64 v2; // r12

  sub_2E5ED70((__int64)rwlock);
  sub_2E6D3E0((__int64)rwlock);
  v1 = sub_22077B0(0x38u);
  v2 = v1;
  if ( v1 )
  {
    *(_QWORD *)(v1 + 8) = 32;
    *(_QWORD *)v1 = "Machine Uniformity Info Analysis";
    *(_QWORD *)(v1 + 16) = "machine-uniformity";
    *(_QWORD *)(v1 + 24) = 18;
    *(_QWORD *)(v1 + 32) = &unk_503F08C;
    *(_WORD *)(v1 + 40) = 256;
    *(_QWORD *)(v1 + 48) = sub_3575090;
  }
  sub_BC3090(rwlock, (_QWORD *)v1, 1);
  return v2;
}
