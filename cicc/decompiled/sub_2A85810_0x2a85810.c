// Function: sub_2A85810
// Address: 0x2a85810
//
__int64 __fastcall sub_2A85810(pthread_rwlock_t *rwlock)
{
  __int64 v1; // rax
  __int64 v2; // r12

  v1 = sub_22077B0(0x38u);
  v2 = v1;
  if ( v1 )
  {
    *(_QWORD *)(v1 + 8) = 25;
    *(_QWORD *)v1 = "Unify function exit nodes";
    *(_QWORD *)(v1 + 16) = "mergereturn";
    *(_QWORD *)(v1 + 24) = 11;
    *(_QWORD *)(v1 + 32) = &unk_500C104;
    *(_WORD *)(v1 + 40) = 0;
    *(_QWORD *)(v1 + 48) = sub_2A85B80;
  }
  sub_BC3090(rwlock, (_QWORD *)v1, 1);
  return v2;
}
