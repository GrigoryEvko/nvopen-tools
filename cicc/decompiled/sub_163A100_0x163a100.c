// Function: sub_163A100
// Address: 0x163a100
//
__int64 sub_163A100()
{
  __int64 v0; // r12

  v0 = sub_22077B0(128);
  if ( v0 )
  {
    memset((void *)v0, 0, 0x80u);
    sub_16C8FC0(v0);
    *(_QWORD *)(v0 + 8) = 0;
    *(_QWORD *)(v0 + 16) = 0;
    *(_QWORD *)(v0 + 24) = 0;
    *(_QWORD *)(v0 + 32) = 0;
    *(_DWORD *)(v0 + 40) = 0;
    *(_QWORD *)(v0 + 48) = 0;
    *(_QWORD *)(v0 + 56) = 0;
    *(_QWORD *)(v0 + 64) = 0x1000000000LL;
    *(_QWORD *)(v0 + 80) = 0;
    *(_QWORD *)(v0 + 88) = 0;
    *(_QWORD *)(v0 + 96) = 0;
    *(_QWORD *)(v0 + 104) = 0;
    *(_QWORD *)(v0 + 112) = 0;
    *(_QWORD *)(v0 + 120) = 0;
  }
  return v0;
}
