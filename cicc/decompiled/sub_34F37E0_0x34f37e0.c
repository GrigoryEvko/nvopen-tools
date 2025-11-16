// Function: sub_34F37E0
// Address: 0x34f37e0
//
__int64 __fastcall sub_34F37E0(pthread_rwlock_t *rwlock)
{
  __int64 v1; // rax
  __int64 v2; // r12

  v1 = sub_22077B0(0x38u);
  v2 = v1;
  if ( v1 )
  {
    *(_QWORD *)(v1 + 8) = 15;
    *(_QWORD *)v1 = "Init Undef Pass";
    *(_QWORD *)(v1 + 16) = "init-undef";
    *(_QWORD *)(v1 + 32) = &unk_503BCAC;
    *(_WORD *)(v1 + 40) = 0;
    *(_QWORD *)(v1 + 24) = 10;
    *(_QWORD *)(v1 + 48) = sub_34F3670;
  }
  sub_BC3090(rwlock, (_QWORD *)v1, 1);
  return v2;
}
