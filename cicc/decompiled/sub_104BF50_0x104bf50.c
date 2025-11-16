// Function: sub_104BF50
// Address: 0x104bf50
//
__int64 __fastcall sub_104BF50(pthread_rwlock_t *rwlock)
{
  __int64 v1; // rax
  __int64 v2; // r12

  v1 = sub_22077B0(56);
  v2 = v1;
  if ( v1 )
  {
    *(_QWORD *)(v1 + 8) = 32;
    *(_QWORD *)v1 = "Post-Dominator Tree Construction";
    *(_QWORD *)(v1 + 16) = "postdomtree";
    *(_QWORD *)(v1 + 24) = 11;
    *(_QWORD *)(v1 + 32) = &unk_4F8FBD4;
    *(_WORD *)(v1 + 40) = 257;
    *(_QWORD *)(v1 + 48) = sub_104C410;
  }
  sub_BC3090(rwlock, (_QWORD *)v1, 1);
  return v2;
}
