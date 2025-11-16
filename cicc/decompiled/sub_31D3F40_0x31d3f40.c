// Function: sub_31D3F40
// Address: 0x31d3f40
//
__int64 __fastcall sub_31D3F40(pthread_rwlock_t *rwlock)
{
  __int64 v1; // rax
  __int64 v2; // r12

  v1 = sub_22077B0(0x38u);
  v2 = v1;
  if ( v1 )
  {
    *(_QWORD *)(v1 + 8) = 39;
    *(_QWORD *)v1 = "Add !range metadata to NVVM intrinsics.";
    *(_QWORD *)(v1 + 16) = "nvvm-intr-range";
    *(_QWORD *)(v1 + 32) = &unk_5035D78;
    *(_WORD *)(v1 + 40) = 0;
    *(_QWORD *)(v1 + 24) = 15;
    *(_QWORD *)(v1 + 48) = sub_31D4640;
  }
  sub_BC3090(rwlock, (_QWORD *)v1, 1);
  return v2;
}
