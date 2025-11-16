// Function: sub_D1A4D0
// Address: 0xd1a4d0
//
__int64 __fastcall sub_D1A4D0(pthread_rwlock_t *rwlock)
{
  __int64 v1; // rax
  __int64 v2; // r12

  sub_D108B0((__int64)rwlock);
  sub_97FFF0((__int64)rwlock);
  v1 = sub_22077B0(56);
  v2 = v1;
  if ( v1 )
  {
    *(_QWORD *)(v1 + 8) = 22;
    *(_QWORD *)v1 = "Globals Alias Analysis";
    *(_QWORD *)(v1 + 16) = "globals-aa";
    *(_QWORD *)(v1 + 24) = 10;
    *(_QWORD *)(v1 + 32) = &unk_4F86B74;
    *(_WORD *)(v1 + 40) = 256;
    *(_QWORD *)(v1 + 48) = sub_D1DA30;
  }
  sub_BC3090(rwlock, (_QWORD *)v1, 1);
  return v2;
}
