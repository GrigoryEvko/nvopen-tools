// Function: sub_2A30B90
// Address: 0x2a30b90
//
__int64 __fastcall sub_2A30B90(pthread_rwlock_t *rwlock)
{
  __int64 v1; // rax
  __int64 v2; // r12

  sub_CFB980((__int64)rwlock);
  sub_22C1050((__int64)rwlock);
  v1 = sub_22077B0(0x38u);
  v2 = v1;
  if ( v1 )
  {
    *(_QWORD *)(v1 + 8) = 30;
    *(_QWORD *)v1 = "Lower SwitchInst's to branches";
    *(_QWORD *)(v1 + 16) = "lowerswitch";
    *(_QWORD *)(v1 + 32) = &unk_500A97C;
    *(_WORD *)(v1 + 40) = 0;
    *(_QWORD *)(v1 + 24) = 11;
    *(_QWORD *)(v1 + 48) = sub_2A32DD0;
  }
  sub_BC3090(rwlock, (_QWORD *)v1, 1);
  return v2;
}
