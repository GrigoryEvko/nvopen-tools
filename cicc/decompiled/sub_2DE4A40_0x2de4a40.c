// Function: sub_2DE4A40
// Address: 0x2de4a40
//
__int64 __fastcall sub_2DE4A40(pthread_rwlock_t *rwlock)
{
  __int64 v1; // rax
  __int64 v2; // r12

  sub_B1A2E0((__int64)rwlock);
  v1 = sub_22077B0(0x38u);
  v2 = v1;
  if ( v1 )
  {
    *(_QWORD *)(v1 + 8) = 30;
    *(_QWORD *)v1 = "Expand indirectbr instructions";
    *(_QWORD *)(v1 + 16) = "indirectbr-expand";
    *(_QWORD *)(v1 + 32) = &unk_501E74C;
    *(_WORD *)(v1 + 40) = 0;
    *(_QWORD *)(v1 + 24) = 17;
    *(_QWORD *)(v1 + 48) = sub_2DE6770;
  }
  sub_BC3090(rwlock, (_QWORD *)v1, 1);
  return v2;
}
