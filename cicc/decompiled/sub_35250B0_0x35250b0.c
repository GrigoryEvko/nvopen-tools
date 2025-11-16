// Function: sub_35250B0
// Address: 0x35250b0
//
__int64 __fastcall sub_35250B0(pthread_rwlock_t *rwlock)
{
  __int64 v1; // rax
  __int64 v2; // r12

  v1 = sub_22077B0(0x38u);
  v2 = v1;
  if ( v1 )
  {
    *(_QWORD *)(v1 + 8) = 24;
    *(_QWORD *)v1 = "Machine CFG Printer Pass";
    *(_QWORD *)(v1 + 16) = "dot-machine-cfg";
    *(_QWORD *)(v1 + 32) = &unk_503CF4C;
    *(_WORD *)(v1 + 40) = 256;
    *(_QWORD *)(v1 + 24) = 15;
    *(_QWORD *)(v1 + 48) = sub_3525360;
  }
  sub_BC3090(rwlock, (_QWORD *)v1, 1);
  return v2;
}
