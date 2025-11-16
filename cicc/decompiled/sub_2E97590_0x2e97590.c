// Function: sub_2E97590
// Address: 0x2e97590
//
__int64 __fastcall sub_2E97590(pthread_rwlock_t *rwlock)
{
  __int64 v1; // rax
  __int64 v2; // r12

  sub_2EA61A0(rwlock);
  sub_2E399F0((__int64)rwlock);
  sub_2E6D3E0((__int64)rwlock);
  sub_CF6DB0((__int64)rwlock);
  v1 = sub_22077B0(0x38u);
  v2 = v1;
  if ( v1 )
  {
    *(_QWORD *)(v1 + 8) = 40;
    *(_QWORD *)v1 = "Early Machine Loop Invariant Code Motion";
    *(_QWORD *)(v1 + 16) = "early-machinelicm";
    *(_QWORD *)(v1 + 32) = &unk_50201E8;
    *(_WORD *)(v1 + 40) = 0;
    *(_QWORD *)(v1 + 24) = 17;
    *(_QWORD *)(v1 + 48) = sub_2E98EC0;
  }
  sub_BC3090(rwlock, (_QWORD *)v1, 1);
  return v2;
}
