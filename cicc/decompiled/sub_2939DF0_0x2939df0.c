// Function: sub_2939DF0
// Address: 0x2939df0
//
__int64 __fastcall sub_2939DF0(pthread_rwlock_t *rwlock)
{
  __int64 v1; // rax
  __int64 v2; // r12

  sub_B1A2E0((__int64)rwlock);
  v1 = sub_22077B0(0x38u);
  v2 = v1;
  if ( v1 )
  {
    *(_QWORD *)(v1 + 8) = 27;
    *(_QWORD *)v1 = "Scalarize vector operations";
    *(_QWORD *)(v1 + 16) = "scalarizer";
    *(_QWORD *)(v1 + 32) = &unk_5005714;
    *(_WORD *)(v1 + 40) = 0;
    *(_QWORD *)(v1 + 24) = 10;
    *(_QWORD *)(v1 + 48) = sub_293A3F0;
  }
  sub_BC3090(rwlock, (_QWORD *)v1, 1);
  return v2;
}
