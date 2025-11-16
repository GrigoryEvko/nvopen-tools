// Function: sub_C53DA0
// Address: 0xc53da0
//
__int64 sub_C53DA0()
{
  __int64 v0; // rax
  __int64 v1; // r12
  __int64 v2; // rax

  v0 = sub_22077B0(352);
  v1 = v0;
  if ( v0 )
  {
    *(_BYTE *)(v0 + 16) = 0;
    *(_QWORD *)v0 = v0 + 16;
    *(_QWORD *)(v0 + 72) = v0 + 88;
    *(_QWORD *)(v0 + 80) = 0x400000000LL;
    *(_QWORD *)(v0 + 128) = v0 + 152;
    *(_QWORD *)(v0 + 8) = 0;
    *(_QWORD *)(v0 + 32) = 0;
    *(_QWORD *)(v0 + 40) = 0;
    *(_QWORD *)(v0 + 48) = 0;
    *(_QWORD *)(v0 + 56) = 0;
    *(_QWORD *)(v0 + 64) = 0;
    *(_QWORD *)(v0 + 120) = 0;
    *(_QWORD *)(v0 + 136) = 16;
    *(_DWORD *)(v0 + 144) = 0;
    *(_BYTE *)(v0 + 148) = 1;
    *(_QWORD *)(v0 + 280) = 0;
    *(_QWORD *)(v0 + 288) = v0 + 312;
    *(_QWORD *)(v0 + 296) = 4;
    *(_DWORD *)(v0 + 304) = 0;
    *(_BYTE *)(v0 + 308) = 1;
    *(_QWORD *)(v0 + 344) = 0;
    v2 = sub_C52570();
    sub_C53C20(v1, v2);
  }
  return v1;
}
