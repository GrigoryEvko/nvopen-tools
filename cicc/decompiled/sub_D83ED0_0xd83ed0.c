// Function: sub_D83ED0
// Address: 0xd83ed0
//
__int64 __fastcall sub_D83ED0(pthread_rwlock_t *rwlock)
{
  __int64 v1; // rax
  __int64 v2; // r12

  v1 = sub_22077B0(56);
  v2 = v1;
  if ( v1 )
  {
    *(_QWORD *)(v1 + 8) = 20;
    *(_QWORD *)v1 = "Profile summary info";
    *(_QWORD *)(v1 + 16) = "profile-summary-info";
    *(_QWORD *)(v1 + 24) = 20;
    *(_QWORD *)(v1 + 32) = &unk_4F87C64;
    *(_WORD *)(v1 + 40) = 256;
    *(_QWORD *)(v1 + 48) = sub_D84AB0;
  }
  sub_BC3090(rwlock, (_QWORD *)v1, 1);
  return v2;
}
