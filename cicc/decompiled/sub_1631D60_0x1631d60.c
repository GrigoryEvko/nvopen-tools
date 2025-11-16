// Function: sub_1631D60
// Address: 0x1631d60
//
_QWORD *__fastcall sub_1631D60(__int64 a1, _BYTE *a2, __int64 a3, __int64 a4)
{
  __int64 v4; // r14
  __int64 v5; // r13
  __int64 v6; // rax
  __int64 v7; // r12
  _QWORD *v8; // rax

  v4 = a1 + 224;
  *(_QWORD *)a1 = a4;
  *(_QWORD *)(a1 + 16) = a1 + 8;
  *(_QWORD *)(a1 + 8) = (a1 + 8) | 4;
  *(_QWORD *)(a1 + 32) = a1 + 24;
  *(_QWORD *)(a1 + 24) = (a1 + 24) | 4;
  *(_QWORD *)(a1 + 48) = a1 + 40;
  *(_QWORD *)(a1 + 40) = (a1 + 40) | 4;
  *(_QWORD *)(a1 + 64) = a1 + 56;
  *(_QWORD *)(a1 + 56) = (a1 + 56) | 4;
  *(_QWORD *)(a1 + 80) = a1 + 72;
  *(_QWORD *)(a1 + 88) = a1 + 104;
  *(_QWORD *)(a1 + 144) = 0x1800000000LL;
  *(_QWORD *)(a1 + 72) = (a1 + 72) | 4;
  *(_QWORD *)(a1 + 96) = 0;
  *(_BYTE *)(a1 + 104) = 0;
  *(_QWORD *)(a1 + 128) = 0;
  *(_QWORD *)(a1 + 136) = 0;
  *(_QWORD *)(a1 + 160) = 0;
  *(_QWORD *)(a1 + 168) = 0;
  *(_QWORD *)(a1 + 176) = a1 + 192;
  if ( a2 )
  {
    v5 = (__int64)&a2[a3];
    sub_1631AB0((__int64 *)(a1 + 176), a2, (__int64)&a2[a3]);
    *(_QWORD *)(a1 + 208) = v4;
    sub_1631AB0((__int64 *)(a1 + 208), a2, v5);
  }
  else
  {
    *(_QWORD *)(a1 + 184) = 0;
    *(_BYTE *)(a1 + 192) = 0;
    *(_QWORD *)(a1 + 208) = v4;
    *(_QWORD *)(a1 + 216) = 0;
    *(_BYTE *)(a1 + 224) = 0;
  }
  *(_QWORD *)(a1 + 248) = 0;
  *(_QWORD *)(a1 + 328) = a1 + 344;
  *(_QWORD *)(a1 + 240) = a1 + 256;
  *(_QWORD *)(a1 + 472) = a1 + 488;
  *(_QWORD *)(a1 + 304) = a1 + 320;
  *(_QWORD *)(a1 + 504) = a1 + 520;
  *(_QWORD *)(a1 + 312) = 0x800000000LL;
  *(_QWORD *)(a1 + 336) = 0x1000000000LL;
  *(_QWORD *)(a1 + 512) = 0x800000000LL;
  *(_QWORD *)(a1 + 688) = a1 + 704;
  *(_QWORD *)(a1 + 696) = 0x800000000LL;
  *(_BYTE *)(a1 + 256) = 0;
  *(_QWORD *)(a1 + 480) = 0;
  *(_BYTE *)(a1 + 488) = 0;
  *(_QWORD *)(a1 + 680) = 0;
  sub_15A9300(a1 + 280, (__int8 *)byte_3F871B3, 0);
  v6 = sub_22077B0(40);
  v7 = v6;
  if ( v6 )
  {
    sub_16D1950(v6, 0, 16);
    *(_DWORD *)(v7 + 32) = 0;
  }
  *(_QWORD *)(a1 + 120) = v7;
  v8 = (_QWORD *)sub_22077B0(32);
  if ( v8 )
  {
    *v8 = 0;
    v8[1] = 0;
    v8[2] = 0x1000000000LL;
  }
  *(_QWORD *)(a1 + 272) = v8;
  return sub_1602610(*(__int64 **)a1, a1);
}
