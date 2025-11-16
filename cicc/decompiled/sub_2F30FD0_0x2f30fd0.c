// Function: sub_2F30FD0
// Address: 0x2f30fd0
//
__int64 __fastcall sub_2F30FD0(pthread_rwlock_t *rwlock)
{
  __int64 v1; // rax
  __int64 v2; // r12

  sub_2E24C30((__int64)rwlock);
  v1 = sub_22077B0(0x38u);
  v2 = v1;
  if ( v1 )
  {
    *(_QWORD *)(v1 + 8) = 43;
    *(_QWORD *)v1 = "Eliminate PHI nodes for register allocation";
    *(_QWORD *)(v1 + 16) = "phi-node-elimination";
    *(_QWORD *)(v1 + 32) = &unk_5022C2C;
    *(_WORD *)(v1 + 40) = 0;
    *(_QWORD *)(v1 + 24) = 20;
    *(_QWORD *)(v1 + 48) = sub_2F320C0;
  }
  sub_BC3090(rwlock, (_QWORD *)v1, 1);
  return v2;
}
