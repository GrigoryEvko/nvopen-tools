// Function: sub_2E91A10
// Address: 0x2e91a10
//
__int64 __fastcall sub_2E91A10(pthread_rwlock_t *rwlock)
{
  __int64 v1; // rax
  __int64 v2; // r12

  v1 = sub_22077B0(0x38u);
  v2 = v1;
  if ( v1 )
  {
    *(_QWORD *)(v1 + 8) = 36;
    *(_QWORD *)v1 = "Finalize machine instruction bundles";
    *(_QWORD *)(v1 + 16) = "finalize-mi-bundles";
    *(_QWORD *)(v1 + 32) = &unk_50201CC;
    *(_WORD *)(v1 + 40) = 0;
    *(_QWORD *)(v1 + 24) = 19;
    *(_QWORD *)(v1 + 48) = sub_2E92290;
  }
  sub_BC3090(rwlock, (_QWORD *)v1, 1);
  return v2;
}
