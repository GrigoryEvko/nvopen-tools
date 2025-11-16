// Function: sub_2E6C080
// Address: 0x2e6c080
//
__int64 __fastcall sub_2E6C080(pthread_rwlock_t *rwlock)
{
  __int64 v1; // rax
  __int64 v2; // r12

  v1 = sub_22077B0(0x38u);
  v2 = v1;
  if ( v1 )
  {
    *(_QWORD *)(v1 + 8) = 34;
    *(_QWORD *)v1 = "MachineDominator Tree Construction";
    *(_QWORD *)(v1 + 16) = "machinedomtree";
    *(_QWORD *)(v1 + 24) = 14;
    *(_QWORD *)(v1 + 32) = &unk_501FE44;
    *(_WORD *)(v1 + 40) = 257;
    *(_QWORD *)(v1 + 48) = sub_2E6D570;
  }
  sub_BC3090(rwlock, (_QWORD *)v1, 1);
  return v2;
}
