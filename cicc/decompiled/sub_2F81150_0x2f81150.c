// Function: sub_2F81150
// Address: 0x2f81150
//
__int64 __fastcall sub_2F81150(pthread_rwlock_t *rwlock)
{
  __int64 v1; // rax
  __int64 v2; // r12

  sub_97FFF0((__int64)rwlock);
  v1 = sub_22077B0(0x38u);
  v2 = v1;
  if ( v1 )
  {
    *(_QWORD *)(v1 + 8) = 47;
    *(_QWORD *)v1 = "Replace intrinsics with calls to vector library";
    *(_QWORD *)(v1 + 16) = "replace-with-veclib";
    *(_QWORD *)(v1 + 24) = 19;
    *(_QWORD *)(v1 + 32) = &unk_5024F68;
    *(_WORD *)(v1 + 40) = 0;
    *(_QWORD *)(v1 + 48) = sub_2F816C0;
  }
  sub_BC3090(rwlock, (_QWORD *)v1, 1);
  return v2;
}
