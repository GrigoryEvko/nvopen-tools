// Function: sub_2A36A40
// Address: 0x2a36a40
//
__int64 __fastcall sub_2A36A40(pthread_rwlock_t *rwlock)
{
  __int64 v1; // rax
  __int64 v2; // r12

  sub_CFB980((__int64)rwlock);
  sub_B1A2E0((__int64)rwlock);
  v1 = sub_22077B0(0x38u);
  v2 = v1;
  if ( v1 )
  {
    *(_QWORD *)(v1 + 8) = 26;
    *(_QWORD *)v1 = "Promote Memory to Register";
    *(_QWORD *)(v1 + 16) = "mem2reg";
    *(_QWORD *)(v1 + 32) = &unk_500A984;
    *(_WORD *)(v1 + 40) = 0;
    *(_QWORD *)(v1 + 24) = 7;
    *(_QWORD *)(v1 + 48) = sub_2A36BF0;
  }
  sub_BC3090(rwlock, (_QWORD *)v1, 1);
  return v2;
}
