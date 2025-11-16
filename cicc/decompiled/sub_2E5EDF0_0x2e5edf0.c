// Function: sub_2E5EDF0
// Address: 0x2e5edf0
//
__int64 __fastcall sub_2E5EDF0(pthread_rwlock_t *rwlock)
{
  __int64 v1; // rax
  __int64 v2; // r12

  sub_2E5ED70((__int64)rwlock);
  v1 = sub_22077B0(0x38u);
  v2 = v1;
  if ( v1 )
  {
    *(_QWORD *)(v1 + 8) = 33;
    *(_QWORD *)v1 = "Print Machine Cycle Info Analysis";
    *(_QWORD *)(v1 + 16) = "print-machine-cycles";
    *(_QWORD *)(v1 + 32) = &unk_501FE2C;
    *(_WORD *)(v1 + 40) = 257;
    *(_QWORD *)(v1 + 24) = 20;
    *(_QWORD *)(v1 + 48) = sub_2E5F0B0;
  }
  sub_BC3090(rwlock, (_QWORD *)v1, 1);
  return v2;
}
