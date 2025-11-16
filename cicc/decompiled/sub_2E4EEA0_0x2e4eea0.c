// Function: sub_2E4EEA0
// Address: 0x2e4eea0
//
__int64 __fastcall sub_2E4EEA0(pthread_rwlock_t *rwlock)
{
  __int64 v1; // rax
  __int64 v2; // r12

  sub_307BEB0();
  sub_DFEA20((__int64)rwlock);
  sub_2E6D3E0(rwlock);
  v1 = sub_22077B0(0x38u);
  v2 = v1;
  if ( v1 )
  {
    *(_QWORD *)(v1 + 8) = 40;
    *(_QWORD *)v1 = "Machine Common Subexpression Elimination";
    *(_QWORD *)(v1 + 16) = "machine-cse";
    *(_QWORD *)(v1 + 32) = &unk_501F54C;
    *(_WORD *)(v1 + 40) = 0;
    *(_QWORD *)(v1 + 24) = 11;
    *(_QWORD *)(v1 + 48) = sub_2E503D0;
  }
  sub_BC3090(rwlock, (_QWORD *)v1, 1);
  return v2;
}
