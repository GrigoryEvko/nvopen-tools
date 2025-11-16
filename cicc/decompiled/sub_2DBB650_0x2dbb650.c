// Function: sub_2DBB650
// Address: 0x2dbb650
//
__int64 __fastcall sub_2DBB650(pthread_rwlock_t *rwlock)
{
  __int64 v1; // rax
  __int64 v2; // r12

  v1 = sub_22077B0(0x38u);
  v2 = v1;
  if ( v1 )
  {
    *(_QWORD *)(v1 + 8) = 23;
    *(_QWORD *)v1 = "Expand large fp convert";
    *(_QWORD *)(v1 + 16) = "expand-large-fp-convert";
    *(_QWORD *)(v1 + 32) = &unk_501D2EC;
    *(_WORD *)(v1 + 40) = 0;
    *(_QWORD *)(v1 + 24) = 23;
    *(_QWORD *)(v1 + 48) = sub_2DBBB20;
  }
  sub_BC3090(rwlock, (_QWORD *)v1, 1);
  return v2;
}
