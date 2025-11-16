// Function: sub_22AB4B0
// Address: 0x22ab4b0
//
__int64 __fastcall sub_22AB4B0(pthread_rwlock_t *rwlock)
{
  __int64 v1; // rax
  __int64 v2; // r12

  sub_CFB980((__int64)rwlock);
  sub_D4AA90((__int64)rwlock);
  sub_B1A2E0((__int64)rwlock);
  sub_D9A960((__int64)rwlock);
  v1 = sub_22077B0(0x38u);
  v2 = v1;
  if ( v1 )
  {
    *(_QWORD *)(v1 + 8) = 24;
    *(_QWORD *)v1 = "Induction Variable Users";
    *(_QWORD *)(v1 + 16) = "iv-users";
    *(_QWORD *)(v1 + 24) = 8;
    *(_QWORD *)(v1 + 32) = &unk_4FDB6AC;
    *(_WORD *)(v1 + 40) = 256;
    *(_QWORD *)(v1 + 48) = sub_22ACD90;
  }
  sub_BC3090(rwlock, (_QWORD *)v1, 1);
  return v2;
}
