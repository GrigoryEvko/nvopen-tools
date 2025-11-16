// Function: sub_2E2CCB0
// Address: 0x2e2ccb0
//
__int64 __fastcall sub_2E2CCB0(pthread_rwlock_t *rwlock)
{
  __int64 v1; // rax
  __int64 v2; // r12

  v1 = sub_22077B0(0x38u);
  v2 = v1;
  if ( v1 )
  {
    *(_QWORD *)(v1 + 8) = 27;
    *(_QWORD *)v1 = "Local Stack Slot Allocation";
    *(_QWORD *)(v1 + 16) = "localstackalloc";
    *(_QWORD *)(v1 + 32) = &unk_501EB24;
    *(_WORD *)(v1 + 40) = 0;
    *(_QWORD *)(v1 + 24) = 15;
    *(_QWORD *)(v1 + 48) = sub_2E2D6F0;
  }
  sub_BC3090(rwlock, (_QWORD *)v1, 1);
  return v2;
}
