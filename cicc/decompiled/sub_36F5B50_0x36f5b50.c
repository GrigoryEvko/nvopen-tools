// Function: sub_36F5B50
// Address: 0x36f5b50
//
__int64 __fastcall sub_36F5B50(pthread_rwlock_t *rwlock)
{
  __int64 v1; // rax
  __int64 v2; // r12

  v1 = sub_22077B0(0x38u);
  v2 = v1;
  if ( v1 )
  {
    *(_QWORD *)(v1 + 8) = 22;
    *(_QWORD *)v1 = "NVPTX ProxyReg Erasure";
    *(_QWORD *)(v1 + 16) = "nvptx-proxyreg-erasure";
    *(_QWORD *)(v1 + 32) = &unk_5041070;
    *(_WORD *)(v1 + 40) = 0;
    *(_QWORD *)(v1 + 24) = 22;
    *(_QWORD *)(v1 + 48) = sub_36F5CC0;
  }
  sub_BC3090(rwlock, (_QWORD *)v1, 1);
  return v2;
}
