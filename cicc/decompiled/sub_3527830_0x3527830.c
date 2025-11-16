// Function: sub_3527830
// Address: 0x3527830
//
__int64 __fastcall sub_3527830(pthread_rwlock_t *rwlock)
{
  __int64 v1; // rax
  __int64 v2; // r12

  v1 = sub_22077B0(0x38u);
  v2 = v1;
  if ( v1 )
  {
    *(_QWORD *)(v1 + 8) = 26;
    *(_QWORD *)v1 = "Machine Check Debug Module";
    *(_QWORD *)(v1 + 16) = "mir-check-debugify";
    *(_QWORD *)(v1 + 32) = &unk_503D244;
    *(_WORD *)(v1 + 40) = 0;
    *(_QWORD *)(v1 + 24) = 18;
    *(_QWORD *)(v1 + 48) = sub_3527750;
  }
  sub_BC3090(rwlock, (_QWORD *)v1, 1);
  return v2;
}
