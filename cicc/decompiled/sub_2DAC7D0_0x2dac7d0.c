// Function: sub_2DAC7D0
// Address: 0x2dac7d0
//
__int64 __fastcall sub_2DAC7D0(pthread_rwlock_t *rwlock)
{
  __int64 v1; // rax
  __int64 v2; // r12

  v1 = sub_22077B0(0x38u);
  v2 = v1;
  if ( v1 )
  {
    *(_QWORD *)(v1 + 8) = 32;
    *(_QWORD *)v1 = "Remove dead machine instructions";
    *(_QWORD *)(v1 + 16) = "dead-mi-elimination";
    *(_QWORD *)(v1 + 32) = &unk_501CF4C;
    *(_WORD *)(v1 + 40) = 0;
    *(_QWORD *)(v1 + 24) = 19;
    *(_QWORD *)(v1 + 48) = sub_2DACA20;
  }
  sub_BC3090(rwlock, (_QWORD *)v1, 1);
  return v2;
}
