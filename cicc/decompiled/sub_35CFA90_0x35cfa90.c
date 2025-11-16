// Function: sub_35CFA90
// Address: 0x35cfa90
//
__int64 __fastcall sub_35CFA90(pthread_rwlock_t *rwlock)
{
  __int64 v1; // rax
  __int64 v2; // r12

  v1 = sub_22077B0(0x38u);
  v2 = v1;
  if ( v1 )
  {
    *(_QWORD *)(v1 + 8) = 18;
    *(_QWORD *)v1 = "Stack Frame Layout";
    *(_QWORD *)(v1 + 16) = "stack-frame-layout";
    *(_QWORD *)(v1 + 32) = &unk_504010C;
    *(_WORD *)(v1 + 40) = 0;
    *(_QWORD *)(v1 + 24) = 18;
    *(_QWORD *)(v1 + 48) = sub_35CFD00;
  }
  sub_BC3090(rwlock, (_QWORD *)v1, 1);
  return v2;
}
