// Function: sub_2DAFAB0
// Address: 0x2dafab0
//
__int64 __fastcall sub_2DAFAB0(pthread_rwlock_t *rwlock)
{
  __int64 v1; // rax
  __int64 v2; // r12

  sub_B1A2E0((__int64)rwlock);
  sub_2FEF6D0(rwlock);
  sub_DFEA20((__int64)rwlock);
  sub_10564E0((__int64)rwlock);
  v1 = sub_22077B0(0x38u);
  v2 = v1;
  if ( v1 )
  {
    *(_QWORD *)(v1 + 8) = 24;
    *(_QWORD *)v1 = "Prepare DWARF exceptions";
    *(_QWORD *)(v1 + 16) = "dwarf-eh-prepare";
    *(_QWORD *)(v1 + 32) = &unk_501CF5C;
    *(_WORD *)(v1 + 40) = 0;
    *(_QWORD *)(v1 + 24) = 16;
    *(_QWORD *)(v1 + 48) = sub_2DAFFA0;
  }
  sub_BC3090(rwlock, (_QWORD *)v1, 1);
  return v2;
}
