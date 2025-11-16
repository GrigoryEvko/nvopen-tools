// Function: sub_2EE73C0
// Address: 0x2ee73c0
//
__int64 __fastcall sub_2EE73C0(pthread_rwlock_t *rwlock)
{
  __int64 v1; // rax
  __int64 v2; // r12

  sub_2EA61A0((__int64)rwlock);
  v1 = sub_22077B0(0x38u);
  v2 = v1;
  if ( v1 )
  {
    *(_QWORD *)(v1 + 8) = 21;
    *(_QWORD *)v1 = "Machine Trace Metrics";
    *(_QWORD *)(v1 + 16) = "machine-trace-metrics";
    *(_QWORD *)(v1 + 24) = 21;
    *(_QWORD *)(v1 + 32) = &unk_502234C;
    *(_WORD *)(v1 + 40) = 256;
    *(_QWORD *)(v1 + 48) = sub_2EE7E10;
  }
  sub_BC3090(rwlock, (_QWORD *)v1, 1);
  return v2;
}
