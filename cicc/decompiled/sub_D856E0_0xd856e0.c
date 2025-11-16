// Function: sub_D856E0
// Address: 0xd856e0
//
__int64 __fastcall sub_D856E0(pthread_rwlock_t *rwlock)
{
  __int64 v1; // rax
  __int64 v2; // r12

  sub_D9A960();
  v1 = sub_22077B0(56);
  v2 = v1;
  if ( v1 )
  {
    *(_QWORD *)(v1 + 8) = 27;
    *(_QWORD *)v1 = "Stack Safety Local Analysis";
    *(_QWORD *)(v1 + 16) = "stack-safety-local";
    *(_QWORD *)(v1 + 24) = 18;
    *(_QWORD *)(v1 + 32) = &unk_4F87F20;
    *(_WORD *)(v1 + 40) = 256;
    *(_QWORD *)(v1 + 48) = sub_D8A180;
  }
  sub_BC3090(rwlock, (_QWORD *)v1, 1);
  return v2;
}
