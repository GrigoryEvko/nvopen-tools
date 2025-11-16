// Function: sub_36F42E0
// Address: 0x36f42e0
//
__int64 __fastcall sub_36F42E0(pthread_rwlock_t *rwlock)
{
  __int64 v1; // rax
  __int64 v2; // r12

  v1 = sub_22077B0(0x38u);
  v2 = v1;
  if ( v1 )
  {
    *(_QWORD *)(v1 + 8) = 14;
    *(_QWORD *)v1 = "NVPTX Peephole";
    *(_QWORD *)(v1 + 16) = "nvptx-peephole";
    *(_QWORD *)(v1 + 32) = &unk_5040F8C;
    *(_WORD *)(v1 + 40) = 0;
    *(_QWORD *)(v1 + 24) = 14;
    *(_QWORD *)(v1 + 48) = sub_36F4F60;
  }
  sub_BC3090(rwlock, (_QWORD *)v1, 1);
  return v2;
}
