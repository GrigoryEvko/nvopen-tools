// Function: sub_3528640
// Address: 0x3528640
//
__int64 __fastcall sub_3528640(pthread_rwlock_t *rwlock)
{
  __int64 v1; // rax
  __int64 v2; // r12

  sub_2EA61A0((__int64)rwlock);
  sub_2EE7B50((__int64)rwlock);
  v1 = sub_22077B0(0x38u);
  v2 = v1;
  if ( v1 )
  {
    *(_QWORD *)(v1 + 8) = 20;
    *(_QWORD *)v1 = "Machine InstCombiner";
    *(_QWORD *)(v1 + 16) = "machine-combiner";
    *(_QWORD *)(v1 + 32) = &unk_503D24C;
    *(_WORD *)(v1 + 40) = 0;
    *(_QWORD *)(v1 + 24) = 16;
    *(_QWORD *)(v1 + 48) = sub_352AA80;
  }
  sub_BC3090(rwlock, (_QWORD *)v1, 1);
  return v2;
}
