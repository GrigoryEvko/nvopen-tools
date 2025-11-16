// Function: sub_2E84C40
// Address: 0x2e84c40
//
__int64 __fastcall sub_2E84C40(pthread_rwlock_t *rwlock)
{
  __int64 v1; // rax
  __int64 v2; // r12

  v1 = sub_22077B0(0x38u);
  v2 = v1;
  if ( v1 )
  {
    *(_QWORD *)(v1 + 8) = 24;
    *(_QWORD *)v1 = "Machine Function Printer";
    *(_QWORD *)(v1 + 16) = "machineinstr-printer";
    *(_QWORD *)(v1 + 32) = &unk_50200EC;
    *(_WORD *)(v1 + 40) = 0;
    *(_QWORD *)(v1 + 24) = 20;
    *(_QWORD *)(v1 + 48) = sub_2E84CC0;
  }
  sub_BC3090(rwlock, (_QWORD *)v1, 1);
  return v2;
}
