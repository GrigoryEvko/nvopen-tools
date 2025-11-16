// Function: sub_2CF5AF0
// Address: 0x2cf5af0
//
__int64 __fastcall sub_2CF5AF0(pthread_rwlock_t *rwlock)
{
  __int64 v1; // rax
  __int64 v2; // r12

  v1 = sub_22077B0(0x38u);
  v2 = v1;
  if ( v1 )
  {
    *(_QWORD *)(v1 + 8) = 41;
    *(_QWORD *)v1 = "Check functions with no 64-bit subscripts";
    *(_QWORD *)(v1 + 16) = "check-gep-index";
    *(_QWORD *)(v1 + 32) = &unk_5014630;
    *(_WORD *)(v1 + 40) = 0;
    *(_QWORD *)(v1 + 24) = 15;
    *(_QWORD *)(v1 + 48) = sub_2CF6160;
  }
  sub_BC3090(rwlock, (_QWORD *)v1, 1);
  return v2;
}
