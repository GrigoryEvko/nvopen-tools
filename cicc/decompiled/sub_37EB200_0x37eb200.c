// Function: sub_37EB200
// Address: 0x37eb200
//
__int64 __fastcall sub_37EB200(pthread_rwlock_t *rwlock)
{
  __int64 v1; // rax
  __int64 v2; // r12

  v1 = sub_22077B0(0x38u);
  v2 = v1;
  if ( v1 )
  {
    *(_QWORD *)(v1 + 8) = 53;
    *(_QWORD *)v1 = "Insert symbols at valid longjmp targets for /guard:cf";
    *(_QWORD *)(v1 + 16) = "CFGuardLongjmp";
    *(_QWORD *)(v1 + 32) = &unk_505133C;
    *(_WORD *)(v1 + 40) = 0;
    *(_QWORD *)(v1 + 24) = 14;
    *(_QWORD *)(v1 + 48) = sub_37EB360;
  }
  sub_BC3090(rwlock, (_QWORD *)v1, 1);
  return v2;
}
