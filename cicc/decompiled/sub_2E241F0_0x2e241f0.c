// Function: sub_2E241F0
// Address: 0x2e241f0
//
__int64 __fastcall sub_2E241F0(pthread_rwlock_t *rwlock)
{
  __int64 v1; // rax
  __int64 v2; // r12

  sub_30050E0();
  v1 = sub_22077B0(0x38u);
  v2 = v1;
  if ( v1 )
  {
    *(_QWORD *)(v1 + 8) = 22;
    *(_QWORD *)v1 = "Live Variable Analysis";
    *(_QWORD *)(v1 + 16) = "livevars";
    *(_QWORD *)(v1 + 24) = 8;
    *(_QWORD *)(v1 + 32) = &unk_501EB14;
    *(_WORD *)(v1 + 40) = 0;
    *(_QWORD *)(v1 + 48) = sub_2E24CB0;
  }
  sub_BC3090(rwlock, (_QWORD *)v1, 1);
  return v2;
}
