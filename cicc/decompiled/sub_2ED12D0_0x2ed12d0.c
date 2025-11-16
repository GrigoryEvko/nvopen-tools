// Function: sub_2ED12D0
// Address: 0x2ed12d0
//
__int64 __fastcall sub_2ED12D0(pthread_rwlock_t *rwlock)
{
  __int64 v1; // rax
  __int64 v2; // r12

  v1 = sub_22077B0(0x38u);
  v2 = v1;
  if ( v1 )
  {
    *(_QWORD *)(v1 + 8) = 19;
    *(_QWORD *)v1 = "PostRA Machine Sink";
    *(_QWORD *)(v1 + 16) = "postra-machine-sink";
    *(_QWORD *)(v1 + 32) = &unk_5021D24;
    *(_WORD *)(v1 + 40) = 0;
    *(_QWORD *)(v1 + 24) = 19;
    *(_QWORD *)(v1 + 48) = sub_2ED1BB0;
  }
  sub_BC3090(rwlock, (_QWORD *)v1, 1);
  return v2;
}
