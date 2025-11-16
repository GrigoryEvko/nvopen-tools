// Function: sub_2E43F40
// Address: 0x2e43f40
//
__int64 __fastcall sub_2E43F40(pthread_rwlock_t *rwlock)
{
  __int64 v1; // rax
  __int64 v2; // r12

  v1 = sub_22077B0(0x38u);
  v2 = v1;
  if ( v1 )
  {
    *(_QWORD *)(v1 + 8) = 35;
    *(_QWORD *)v1 = "Machine Branch Probability Analysis";
    *(_QWORD *)(v1 + 16) = "machine-branch-prob";
    *(_QWORD *)(v1 + 24) = 19;
    *(_QWORD *)(v1 + 32) = &unk_501F1C8;
    *(_WORD *)(v1 + 40) = 256;
    *(_QWORD *)(v1 + 48) = sub_2E44190;
  }
  sub_BC3090(rwlock, (_QWORD *)v1, 1);
  return v2;
}
