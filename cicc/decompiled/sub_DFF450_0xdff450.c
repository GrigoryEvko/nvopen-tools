// Function: sub_DFF450
// Address: 0xdff450
//
__int64 __fastcall sub_DFF450(pthread_rwlock_t *rwlock)
{
  __int64 v1; // rax
  __int64 v2; // r12

  v1 = sub_22077B0(56);
  v2 = v1;
  if ( v1 )
  {
    *(_QWORD *)(v1 + 8) = 25;
    *(_QWORD *)v1 = "Type-Based Alias Analysis";
    *(_QWORD *)(v1 + 16) = "tbaa";
    *(_QWORD *)(v1 + 24) = 4;
    *(_QWORD *)(v1 + 32) = &unk_4F89FAC;
    *(_WORD *)(v1 + 40) = 256;
    *(_QWORD *)(v1 + 48) = sub_E006E0;
  }
  sub_BC3090(rwlock, (_QWORD *)v1, 1);
  return v2;
}
