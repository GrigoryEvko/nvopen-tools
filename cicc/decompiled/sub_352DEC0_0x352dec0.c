// Function: sub_352DEC0
// Address: 0x352dec0
//
__int64 __fastcall sub_352DEC0(pthread_rwlock_t *rwlock)
{
  __int64 v1; // rax
  __int64 v2; // r12

  v1 = sub_22077B0(0x38u);
  v2 = v1;
  if ( v1 )
  {
    *(_QWORD *)(v1 + 8) = 23;
    *(_QWORD *)v1 = "Machine Debugify Module";
    *(_QWORD *)(v1 + 16) = "mir-debugify";
    *(_QWORD *)(v1 + 32) = &unk_503D4EC;
    *(_WORD *)(v1 + 40) = 0;
    *(_QWORD *)(v1 + 24) = 12;
    *(_QWORD *)(v1 + 48) = sub_352DDE0;
  }
  sub_BC3090(rwlock, (_QWORD *)v1, 1);
  return v2;
}
