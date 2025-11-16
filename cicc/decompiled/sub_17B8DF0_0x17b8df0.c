// Function: sub_17B8DF0
// Address: 0x17b8df0
//
__int64 sub_17B8DF0()
{
  __int64 v0; // r12
  unsigned __int64 v1; // rax
  __int16 v2; // dx
  __int64 v3; // rax

  v0 = sub_22077B0(376);
  if ( v0 )
  {
    v1 = sub_17B8C30();
    *(_QWORD *)(v0 + 8) = 0;
    *(_QWORD *)(v0 + 160) = v1;
    *(_QWORD *)(v0 + 80) = v0 + 64;
    *(_QWORD *)(v0 + 88) = v0 + 64;
    *(_QWORD *)(v0 + 128) = v0 + 112;
    *(_QWORD *)(v0 + 136) = v0 + 112;
    *(_QWORD *)(v0 + 176) = v0 + 192;
    *(_QWORD *)(v0 + 184) = 0x400000000LL;
    *(_QWORD *)(v0 + 232) = v0 + 248;
    *(_QWORD *)(v0 + 16) = &unk_4FA29AC;
    *(_DWORD *)(v0 + 24) = 5;
    *(_QWORD *)(v0 + 32) = 0;
    *(_QWORD *)(v0 + 40) = 0;
    *(_QWORD *)(v0 + 48) = 0;
    *(_DWORD *)(v0 + 64) = 0;
    *(_QWORD *)(v0 + 72) = 0;
    *(_QWORD *)(v0 + 96) = 0;
    *(_DWORD *)(v0 + 112) = 0;
    *(_QWORD *)(v0 + 120) = 0;
    *(_QWORD *)(v0 + 144) = 0;
    *(_BYTE *)(v0 + 152) = 0;
    *(_QWORD *)v0 = off_49F0128;
    *(_WORD *)(v0 + 168) = v2;
    *(_QWORD *)(v0 + 240) = 0x1000000000LL;
    *(_DWORD *)(v0 + 170) = _byteswap_ulong(v1 >> 16);
    *(_BYTE *)(v0 + 174) = 0;
    v3 = sub_163A1D0();
    sub_17B8D00(v3);
  }
  return v0;
}
