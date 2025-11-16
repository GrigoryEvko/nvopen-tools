// Function: sub_229A720
// Address: 0x229a720
//
__int64 __fastcall sub_229A720(pthread_rwlock_t *rwlock)
{
  __int64 v1; // rax
  __int64 v2; // r12

  v1 = sub_22077B0(0x38u);
  v2 = v1;
  if ( v1 )
  {
    *(_QWORD *)(v1 + 8) = 57;
    *(_QWORD *)v1 = "View dominance tree of function (with no function bodies)";
    *(_QWORD *)(v1 + 16) = "view-dom-only";
    *(_QWORD *)(v1 + 32) = &unk_4FDB61C;
    *(_WORD *)(v1 + 40) = 0;
    *(_QWORD *)(v1 + 24) = 13;
    *(_QWORD *)(v1 + 48) = sub_229BE80;
  }
  sub_BC3090(rwlock, (_QWORD *)v1, 1);
  return v2;
}
