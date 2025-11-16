// Function: sub_35DE0B0
// Address: 0x35de0b0
//
__int64 __fastcall sub_35DE0B0(pthread_rwlock_t *rwlock)
{
  __int64 v1; // rax
  __int64 v2; // r12

  sub_D4AA90((__int64)rwlock);
  sub_2FEF6D0((__int64)rwlock);
  sub_DFEA20((__int64)rwlock);
  v1 = sub_22077B0(0x38u);
  v2 = v1;
  if ( v1 )
  {
    *(_QWORD *)(v1 + 8) = 14;
    *(_QWORD *)v1 = "Type Promotion";
    *(_QWORD *)(v1 + 16) = "type-promotion";
    *(_QWORD *)(v1 + 32) = &unk_50401ED;
    *(_WORD *)(v1 + 40) = 0;
    *(_QWORD *)(v1 + 24) = 14;
    *(_QWORD *)(v1 + 48) = sub_35DE420;
  }
  sub_BC3090(rwlock, (_QWORD *)v1, 1);
  return v2;
}
