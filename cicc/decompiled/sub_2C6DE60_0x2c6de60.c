// Function: sub_2C6DE60
// Address: 0x2c6de60
//
__int64 __fastcall sub_2C6DE60(pthread_rwlock_t *rwlock)
{
  __int64 v1; // rax
  __int64 v2; // r12

  sub_B1A2E0((__int64)rwlock);
  v1 = sub_22077B0(0x38u);
  v2 = v1;
  if ( v1 )
  {
    *(_QWORD *)(v1 + 8) = 10;
    *(_QWORD *)v1 = "Merge sets";
    *(_QWORD *)(v1 + 16) = "merge-sets";
    *(_QWORD *)(v1 + 24) = 10;
    *(_QWORD *)(v1 + 32) = &unk_5010CD4;
    *(_WORD *)(v1 + 40) = 257;
    *(_QWORD *)(v1 + 48) = sub_2C6F300;
  }
  sub_BC3090(rwlock, (_QWORD *)v1, 1);
  return v2;
}
