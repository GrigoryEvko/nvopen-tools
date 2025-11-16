// Function: sub_22A3C40
// Address: 0x22a3c40
//
__int64 __fastcall sub_22A3C40(pthread_rwlock_t *rwlock)
{
  __int64 v1; // rax
  __int64 v2; // r12

  sub_B1A2E0((__int64)rwlock);
  v1 = sub_22077B0(0x38u);
  v2 = v1;
  if ( v1 )
  {
    *(_QWORD *)(v1 + 8) = 31;
    *(_QWORD *)v1 = "Dominance Frontier Construction";
    *(_QWORD *)(v1 + 16) = "domfrontier";
    *(_QWORD *)(v1 + 24) = 11;
    *(_QWORD *)(v1 + 32) = &unk_4FDB684;
    *(_WORD *)(v1 + 40) = 257;
    *(_QWORD *)(v1 + 48) = sub_22A4340;
  }
  sub_BC3090(rwlock, (_QWORD *)v1, 1);
  return v2;
}
