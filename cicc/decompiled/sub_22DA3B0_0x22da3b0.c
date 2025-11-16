// Function: sub_22DA3B0
// Address: 0x22da3b0
//
__int64 __fastcall sub_22DA3B0(pthread_rwlock_t *rwlock)
{
  __int64 v1; // rax
  __int64 v2; // r12

  sub_B1A2E0((__int64)rwlock);
  sub_104C240((__int64)rwlock);
  sub_22A4190((__int64)rwlock);
  v1 = sub_22077B0(0x38u);
  v2 = v1;
  if ( v1 )
  {
    *(_QWORD *)(v1 + 8) = 39;
    *(_QWORD *)v1 = "Detect single entry single exit regions";
    *(_QWORD *)(v1 + 16) = "regions";
    *(_QWORD *)(v1 + 24) = 7;
    *(_QWORD *)(v1 + 32) = &unk_4FDBD0C;
    *(_WORD *)(v1 + 40) = 257;
    *(_QWORD *)(v1 + 48) = sub_22DC4B0;
  }
  sub_BC3090(rwlock, (_QWORD *)v1, 1);
  return v2;
}
