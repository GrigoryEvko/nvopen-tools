// Function: sub_2F25B40
// Address: 0x2f25b40
//
__int64 __fastcall sub_2F25B40(pthread_rwlock_t *rwlock)
{
  __int64 v1; // rax
  __int64 v2; // r12

  sub_2E6D3E0((__int64)rwlock);
  v1 = sub_22077B0(0x38u);
  v2 = v1;
  if ( v1 )
  {
    *(_QWORD *)(v1 + 8) = 33;
    *(_QWORD *)v1 = "Optimize machine instruction PHIs";
    *(_QWORD *)(v1 + 16) = "opt-phis";
    *(_QWORD *)(v1 + 32) = &unk_5022610;
    *(_WORD *)(v1 + 40) = 0;
    *(_QWORD *)(v1 + 24) = 8;
    *(_QWORD *)(v1 + 48) = sub_2F26400;
  }
  sub_BC3090(rwlock, (_QWORD *)v1, 1);
  return v2;
}
