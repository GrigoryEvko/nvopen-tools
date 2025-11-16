// Function: sub_CF4450
// Address: 0xcf4450
//
__int64 __fastcall sub_CF4450(pthread_rwlock_t *rwlock)
{
  __int64 v1; // rax
  __int64 v2; // r12

  v1 = sub_22077B0(56);
  v2 = v1;
  if ( v1 )
  {
    *(_QWORD *)(v1 + 8) = 23;
    *(_QWORD *)v1 = "External Alias Analysis";
    *(_QWORD *)(v1 + 16) = "external-aa";
    *(_QWORD *)(v1 + 24) = 11;
    *(_QWORD *)(v1 + 32) = &unk_4F86538;
    *(_WORD *)(v1 + 40) = 256;
    *(_QWORD *)(v1 + 48) = sub_CF6B10;
  }
  sub_BC3090(rwlock, (_QWORD *)v1, 1);
  return v2;
}
