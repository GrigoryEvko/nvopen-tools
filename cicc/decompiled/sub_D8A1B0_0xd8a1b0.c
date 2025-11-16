// Function: sub_D8A1B0
// Address: 0xd8a1b0
//
__int64 __fastcall sub_D8A1B0(pthread_rwlock_t *rwlock)
{
  __int64 v1; // rax
  __int64 v2; // r12

  sub_D8A010((__int64)rwlock);
  sub_D783E0((__int64)rwlock);
  v1 = sub_22077B0(56);
  v2 = v1;
  if ( v1 )
  {
    *(_QWORD *)(v1 + 8) = 21;
    *(_QWORD *)v1 = "Stack Safety Analysis";
    *(_QWORD *)(v1 + 16) = "stack-safety";
    *(_QWORD *)(v1 + 24) = 12;
    *(_QWORD *)(v1 + 32) = &unk_4F87F10;
    *(_WORD *)(v1 + 40) = 256;
    *(_QWORD *)(v1 + 48) = sub_D8A3B0;
  }
  sub_BC3090(rwlock, (_QWORD *)v1, 1);
  return v2;
}
