// Function: sub_2EEE4C0
// Address: 0x2eee4c0
//
__int64 __fastcall sub_2EEE4C0(pthread_rwlock_t *rwlock)
{
  __int64 v1; // rax
  __int64 v2; // r12

  v1 = sub_22077B0(0x38u);
  v2 = v1;
  if ( v1 )
  {
    *(_QWORD *)(v1 + 8) = 29;
    *(_QWORD *)v1 = "Verify generated machine code";
    *(_QWORD *)(v1 + 16) = "machineverifier";
    *(_QWORD *)(v1 + 32) = &unk_502235C;
    *(_WORD *)(v1 + 40) = 0;
    *(_QWORD *)(v1 + 24) = 15;
    *(_QWORD *)(v1 + 48) = sub_2EF8B80;
  }
  sub_BC3090(rwlock, (_QWORD *)v1, 1);
  return v2;
}
