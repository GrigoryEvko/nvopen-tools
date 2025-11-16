// Function: sub_2FCE8A0
// Address: 0x2fce8a0
//
__int64 __fastcall sub_2FCE8A0(pthread_rwlock_t *rwlock)
{
  __int64 v1; // rax
  __int64 v2; // r12

  sub_2FACF50((__int64)rwlock);
  sub_2E22F70((__int64)rwlock);
  sub_2EA61A0((__int64)rwlock);
  v1 = sub_22077B0(0x38u);
  v2 = v1;
  if ( v1 )
  {
    *(_QWORD *)(v1 + 8) = 19;
    *(_QWORD *)v1 = "Stack Slot Coloring";
    *(_QWORD *)(v1 + 16) = "stack-slot-coloring";
    *(_QWORD *)(v1 + 32) = &unk_502624C;
    *(_WORD *)(v1 + 40) = 0;
    *(_QWORD *)(v1 + 24) = 19;
    *(_QWORD *)(v1 + 48) = sub_2FCFEB0;
  }
  sub_BC3090(rwlock, (_QWORD *)v1, 1);
  return v2;
}
