// Function: sub_35C58E0
// Address: 0x35c58e0
//
__int64 __fastcall sub_35C58E0(pthread_rwlock_t *rwlock)
{
  __int64 v1; // rax
  __int64 v2; // r12

  v1 = sub_22077B0(0x38u);
  v2 = v1;
  if ( v1 )
  {
    *(_QWORD *)(v1 + 8) = 46;
    *(_QWORD *)v1 = "Scavenge virtual registers inside basic blocks";
    *(_QWORD *)(v1 + 16) = "scavenger-test";
    *(_QWORD *)(v1 + 32) = &unk_503FF34;
    *(_WORD *)(v1 + 40) = 0;
    *(_QWORD *)(v1 + 24) = 14;
    *(_QWORD *)(v1 + 48) = sub_35C57D0;
  }
  sub_BC3090(rwlock, (_QWORD *)v1, 1);
  return v2;
}
