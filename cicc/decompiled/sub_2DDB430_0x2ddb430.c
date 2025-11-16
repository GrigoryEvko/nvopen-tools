// Function: sub_2DDB430
// Address: 0x2ddb430
//
__int64 __fastcall sub_2DDB430(pthread_rwlock_t *rwlock)
{
  __int64 v1; // rax
  __int64 v2; // r12

  v1 = sub_22077B0(0x38u);
  v2 = v1;
  if ( v1 )
  {
    *(_QWORD *)(v1 + 8) = 26;
    *(_QWORD *)v1 = "Global merge function pass";
    *(_QWORD *)(v1 + 16) = "global-merge-func";
    *(_QWORD *)(v1 + 32) = &unk_501E12C;
    *(_WORD *)(v1 + 40) = 0;
    *(_QWORD *)(v1 + 24) = 17;
    *(_QWORD *)(v1 + 48) = sub_2DDD1E0;
  }
  sub_BC3090(rwlock, (_QWORD *)v1, 1);
  return v2;
}
