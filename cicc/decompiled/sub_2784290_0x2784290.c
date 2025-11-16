// Function: sub_2784290
// Address: 0x2784290
//
__int64 __fastcall sub_2784290(pthread_rwlock_t *rwlock)
{
  __int64 v1; // rax
  __int64 v2; // r12

  sub_CF6DB0((__int64)rwlock);
  v1 = sub_22077B0(0x38u);
  v2 = v1;
  if ( v1 )
  {
    *(_QWORD *)(v1 + 8) = 15;
    *(_QWORD *)v1 = "Flatten the CFG";
    *(_QWORD *)(v1 + 16) = "flattencfg";
    *(_QWORD *)(v1 + 32) = &unk_4FFB2AC;
    *(_WORD *)(v1 + 40) = 0;
    *(_QWORD *)(v1 + 24) = 10;
    *(_QWORD *)(v1 + 48) = sub_2784410;
  }
  sub_BC3090(rwlock, (_QWORD *)v1, 1);
  return v2;
}
