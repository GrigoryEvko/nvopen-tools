// Function: sub_35AA4B0
// Address: 0x35aa4b0
//
__int64 __fastcall sub_35AA4B0(pthread_rwlock_t *rwlock)
{
  __int64 v1; // rax
  __int64 v2; // r12

  v1 = sub_22077B0(0x38u);
  v2 = v1;
  if ( v1 )
  {
    *(_QWORD *)(v1 + 8) = 25;
    *(_QWORD *)v1 = "Post RA hazard recognizer";
    *(_QWORD *)(v1 + 16) = "post-RA-hazard-rec";
    *(_QWORD *)(v1 + 32) = &unk_503FCEC;
    *(_WORD *)(v1 + 40) = 0;
    *(_QWORD *)(v1 + 24) = 18;
    *(_QWORD *)(v1 + 48) = sub_35AA3A0;
  }
  sub_BC3090(rwlock, (_QWORD *)v1, 1);
  return v2;
}
