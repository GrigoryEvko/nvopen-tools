// Function: sub_28C13A0
// Address: 0x28c13a0
//
__int64 __fastcall sub_28C13A0(pthread_rwlock_t *rwlock)
{
  __int64 v1; // rax
  __int64 v2; // r12

  sub_CFB980((__int64)rwlock);
  sub_B1A2E0((__int64)rwlock);
  sub_D9A960((__int64)rwlock);
  sub_97FFF0((__int64)rwlock);
  sub_DFEA20((__int64)rwlock);
  v1 = sub_22077B0(0x38u);
  v2 = v1;
  if ( v1 )
  {
    *(_QWORD *)(v1 + 8) = 18;
    *(_QWORD *)v1 = "Nary reassociation";
    *(_QWORD *)(v1 + 16) = "nary-reassociate";
    *(_QWORD *)(v1 + 32) = &unk_50044B4;
    *(_WORD *)(v1 + 40) = 0;
    *(_QWORD *)(v1 + 24) = 16;
    *(_QWORD *)(v1 + 48) = sub_28C1A00;
  }
  sub_BC3090(rwlock, (_QWORD *)v1, 1);
  return v2;
}
