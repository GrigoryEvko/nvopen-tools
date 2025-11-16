// Function: sub_B184B0
// Address: 0xb184b0
//
__int64 __fastcall sub_B184B0(pthread_rwlock_t *rwlock)
{
  __int64 v1; // rax
  __int64 v2; // r12

  v1 = sub_22077B0(56);
  v2 = v1;
  if ( v1 )
  {
    *(_QWORD *)(v1 + 8) = 27;
    *(_QWORD *)v1 = "Dominator Tree Construction";
    *(_QWORD *)(v1 + 16) = "domtree";
    *(_QWORD *)(v1 + 24) = 7;
    *(_QWORD *)(v1 + 32) = &unk_4F8144C;
    *(_WORD *)(v1 + 40) = 257;
    *(_QWORD *)(v1 + 48) = sub_B1A4B0;
  }
  sub_BC3090(rwlock);
  return v2;
}
