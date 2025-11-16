// Function: sub_AE41B0
// Address: 0xae41b0
//
__int64 __fastcall sub_AE41B0(__int64 a1, _BYTE *a2, __int64 a3)
{
  unsigned __int64 v4; // rax
  char v5; // al
  unsigned __int64 v7; // [rsp+8h] [rbp-218h] BYREF
  _QWORD v8[66]; // [rsp+10h] [rbp-210h] BYREF

  sub_AE1D50((__int64)v8);
  sub_AE3AA0(&v7, v8, a2, a3);
  v4 = v7 & 0xFFFFFFFFFFFFFFFELL;
  if ( (v7 & 0xFFFFFFFFFFFFFFFELL) != 0 )
  {
    *(_BYTE *)(a1 + 496) |= 3u;
    *(_QWORD *)a1 = v4;
  }
  else
  {
    *(_BYTE *)a1 = 0;
    *(_QWORD *)(a1 + 32) = a1 + 56;
    *(_QWORD *)(a1 + 64) = a1 + 80;
    *(_QWORD *)(a1 + 72) = 0x600000000LL;
    *(_QWORD *)(a1 + 128) = a1 + 144;
    *(_QWORD *)(a1 + 136) = 0x400000000LL;
    *(_QWORD *)(a1 + 176) = a1 + 192;
    *(_QWORD *)(a1 + 184) = 0xA00000000LL;
    *(_QWORD *)(a1 + 272) = a1 + 288;
    *(_QWORD *)(a1 + 280) = 0x800000000LL;
    *(_QWORD *)(a1 + 448) = a1 + 464;
    *(_WORD *)(a1 + 480) = 768;
    v5 = *(_BYTE *)(a1 + 496);
    *(_DWORD *)(a1 + 4) = 0;
    *(_BYTE *)(a1 + 17) = 0;
    *(_BYTE *)(a1 + 27) = 0;
    *(_QWORD *)(a1 + 8) = 0;
    *(_QWORD *)(a1 + 19) = 0;
    *(_QWORD *)(a1 + 40) = 0;
    *(_QWORD *)(a1 + 48) = 8;
    *(_QWORD *)(a1 + 456) = 0;
    *(_BYTE *)(a1 + 464) = 0;
    *(_QWORD *)(a1 + 488) = 0;
    *(_BYTE *)(a1 + 496) = v5 & 0xFC | 2;
    sub_AE1EA0(a1, (__int64)v8);
  }
  sub_AE4030(v8, (__int64)v8);
  return a1;
}
