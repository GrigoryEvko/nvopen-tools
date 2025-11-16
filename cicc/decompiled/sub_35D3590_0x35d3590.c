// Function: sub_35D3590
// Address: 0x35d3590
//
__int64 __fastcall sub_35D3590(pthread_rwlock_t *rwlock)
{
  __int64 v1; // rax
  __int64 v2; // r12

  v1 = sub_22077B0(0x38u);
  v2 = v1;
  if ( v1 )
  {
    *(_QWORD *)(v1 + 8) = 26;
    *(_QWORD *)v1 = "StackMap Liveness Analysis";
    *(_QWORD *)(v1 + 16) = "stackmap-liveness";
    *(_QWORD *)(v1 + 32) = &unk_5040114;
    *(_WORD *)(v1 + 40) = 0;
    *(_QWORD *)(v1 + 24) = 17;
    *(_QWORD *)(v1 + 48) = sub_35D3A40;
  }
  sub_BC3090(rwlock, (_QWORD *)v1, 1);
  return v2;
}
