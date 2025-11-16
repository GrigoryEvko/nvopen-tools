// Function: sub_2E91990
// Address: 0x2e91990
//
__int64 __fastcall sub_2E91990(pthread_rwlock_t *rwlock)
{
  __int64 v1; // rax
  __int64 v2; // r12

  v1 = sub_22077B0(0x38u);
  v2 = v1;
  if ( v1 )
  {
    *(_QWORD *)(v1 + 8) = 34;
    *(_QWORD *)v1 = "Unpack machine instruction bundles";
    *(_QWORD *)(v1 + 16) = "unpack-mi-bundles";
    *(_QWORD *)(v1 + 32) = &unk_50201D4;
    *(_WORD *)(v1 + 40) = 0;
    *(_QWORD *)(v1 + 24) = 17;
    *(_QWORD *)(v1 + 48) = sub_2E92060;
  }
  sub_BC3090(rwlock, (_QWORD *)v1, 1);
  return v2;
}
