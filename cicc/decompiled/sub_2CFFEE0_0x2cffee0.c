// Function: sub_2CFFEE0
// Address: 0x2cffee0
//
__int64 __fastcall sub_2CFFEE0(pthread_rwlock_t *rwlock)
{
  __int64 v1; // rax
  __int64 v2; // r12

  v1 = sub_22077B0(0x38u);
  v2 = v1;
  if ( v1 )
  {
    *(_QWORD *)(v1 + 8) = 31;
    *(_QWORD *)v1 = "Propagate alignment information";
    *(_QWORD *)(v1 + 16) = "nvvm-propagate-alignment";
    *(_QWORD *)(v1 + 24) = 24;
    *(_QWORD *)(v1 + 32) = &unk_5014C4C;
    *(_WORD *)(v1 + 40) = 0;
    *(_QWORD *)(v1 + 48) = sub_2D00900;
  }
  sub_BC3090(rwlock, (_QWORD *)v1, 1);
  return v2;
}
