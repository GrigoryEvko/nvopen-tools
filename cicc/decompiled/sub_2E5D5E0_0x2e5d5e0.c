// Function: sub_2E5D5E0
// Address: 0x2e5d5e0
//
__int64 __fastcall sub_2E5D5E0(pthread_rwlock_t *rwlock)
{
  __int64 v1; // rax
  __int64 v2; // r12

  v1 = sub_22077B0(0x38u);
  v2 = v1;
  if ( v1 )
  {
    *(_QWORD *)(v1 + 8) = 27;
    *(_QWORD *)v1 = "Machine Cycle Info Analysis";
    *(_QWORD *)(v1 + 16) = "machine-cycles";
    *(_QWORD *)(v1 + 24) = 14;
    *(_QWORD *)(v1 + 32) = &unk_501FE3C;
    *(_WORD *)(v1 + 40) = 257;
    *(_QWORD *)(v1 + 48) = sub_2E5F000;
  }
  sub_BC3090(rwlock, (_QWORD *)v1, 1);
  return v2;
}
