// Function: sub_2EB2D70
// Address: 0x2eb2d70
//
__int64 __fastcall sub_2EB2D70(pthread_rwlock_t *rwlock)
{
  __int64 v1; // rax
  __int64 v2; // r12

  v1 = sub_22077B0(0x38u);
  v2 = v1;
  if ( v1 )
  {
    *(_QWORD *)(v1 + 8) = 38;
    *(_QWORD *)v1 = "MachinePostDominator Tree Construction";
    *(_QWORD *)(v1 + 16) = "machinepostdomtree";
    *(_QWORD *)(v1 + 24) = 18;
    *(_QWORD *)(v1 + 32) = &unk_50209DC;
    *(_WORD *)(v1 + 40) = 257;
    *(_QWORD *)(v1 + 48) = sub_2EB40C0;
  }
  sub_BC3090(rwlock, (_QWORD *)v1, 1);
  return v2;
}
