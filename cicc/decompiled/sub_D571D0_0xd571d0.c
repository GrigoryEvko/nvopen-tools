// Function: sub_D571D0
// Address: 0xd571d0
//
__int64 __fastcall sub_D571D0(pthread_rwlock_t *rwlock)
{
  __int64 v1; // rax
  __int64 v2; // r12

  v1 = sub_22077B0(56);
  v2 = v1;
  if ( v1 )
  {
    *(_QWORD *)(v1 + 8) = 14;
    *(_QWORD *)v1 = "LCSSA Verifier";
    *(_QWORD *)(v1 + 16) = "lcssa-verification";
    *(_QWORD *)(v1 + 24) = 18;
    *(_QWORD *)(v1 + 32) = &unk_4F876DC;
    *(_WORD *)(v1 + 40) = 0;
    *(_QWORD *)(v1 + 48) = sub_D58110;
  }
  sub_BC3090(rwlock, (_QWORD *)v1, 1);
  return v2;
}
