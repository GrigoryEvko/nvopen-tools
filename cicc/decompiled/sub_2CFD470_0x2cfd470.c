// Function: sub_2CFD470
// Address: 0x2cfd470
//
__int64 __fastcall sub_2CFD470(pthread_rwlock_t *rwlock)
{
  __int64 v1; // rax
  __int64 v2; // r12

  sub_B1A2E0((__int64)rwlock);
  sub_104C240((__int64)rwlock);
  v1 = sub_22077B0(0x38u);
  v2 = v1;
  if ( v1 )
  {
    *(_QWORD *)(v1 + 8) = 29;
    *(_QWORD *)v1 = "Process __restrict__ keyword.";
    *(_QWORD *)(v1 + 16) = "Process-Restrict";
    *(_QWORD *)(v1 + 32) = &unk_50148CD;
    *(_WORD *)(v1 + 40) = 0;
    *(_QWORD *)(v1 + 24) = 16;
    *(_QWORD *)(v1 + 48) = sub_2CFE020;
  }
  sub_BC3090(rwlock, (_QWORD *)v1, 1);
  return v2;
}
