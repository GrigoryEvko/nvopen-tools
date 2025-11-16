// Function: sub_36D0050
// Address: 0x36d0050
//
__int64 __fastcall sub_36D0050(pthread_rwlock_t *rwlock)
{
  __int64 v1; // rax
  __int64 v2; // r12

  v1 = sub_22077B0(0x38u);
  v2 = v1;
  if ( v1 )
  {
    *(_QWORD *)(v1 + 8) = 67;
    *(_QWORD *)v1 = "Hoisting alloca instructions in non-entry blocks to the entry block";
    *(_QWORD *)(v1 + 16) = "alloca-hoisting";
    *(_QWORD *)(v1 + 32) = &unk_5040BCC;
    *(_WORD *)(v1 + 40) = 0;
    *(_QWORD *)(v1 + 24) = 15;
    *(_QWORD *)(v1 + 48) = sub_36CFF70;
  }
  sub_BC3090(rwlock, (_QWORD *)v1, 1);
  return v2;
}
