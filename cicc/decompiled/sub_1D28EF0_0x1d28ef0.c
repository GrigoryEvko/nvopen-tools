// Function: sub_1D28EF0
// Address: 0x1d28ef0
//
__int64 __fastcall sub_1D28EF0(
        __int64 a1,
        __int16 a2,
        __int64 a3,
        __int64 *a4,
        __int64 a5,
        __int64 a6,
        __int128 a7,
        char a8)
{
  __int64 v11; // rax
  __int64 v12; // rsi
  unsigned __int8 *v13; // rsi
  __int64 v15; // [rsp+8h] [rbp-58h]
  int v16; // [rsp+14h] [rbp-4Ch]
  _QWORD v18[7]; // [rsp+28h] [rbp-38h] BYREF

  v16 = a3;
  v11 = sub_1D274F0(a7, a3, (__int64)a4, a5, a6);
  v12 = *a4;
  v18[0] = v12;
  if ( v12 )
  {
    v15 = v11;
    sub_1623A60((__int64)v18, v12, 2);
    v13 = (unsigned __int8 *)v18[0];
    *(_QWORD *)a1 = 0;
    *(_QWORD *)(a1 + 8) = 0;
    *(_QWORD *)(a1 + 16) = 0;
    *(_WORD *)(a1 + 24) = a2;
    *(_DWORD *)(a1 + 28) = -1;
    *(_QWORD *)(a1 + 32) = 0;
    *(_QWORD *)(a1 + 40) = v15;
    *(_QWORD *)(a1 + 48) = 0;
    *(_QWORD *)(a1 + 56) = 0x100000000LL;
    *(_DWORD *)(a1 + 64) = v16;
    *(_QWORD *)(a1 + 72) = v13;
    if ( v13 )
      sub_1623210((__int64)v18, v13, a1 + 72);
  }
  else
  {
    *(_QWORD *)(a1 + 40) = v11;
    *(_QWORD *)(a1 + 56) = 0x100000000LL;
    *(_QWORD *)a1 = 0;
    *(_QWORD *)(a1 + 8) = 0;
    *(_QWORD *)(a1 + 16) = 0;
    *(_WORD *)(a1 + 24) = a2;
    *(_DWORD *)(a1 + 28) = -1;
    *(_QWORD *)(a1 + 32) = 0;
    *(_QWORD *)(a1 + 48) = 0;
    *(_DWORD *)(a1 + 64) = v16;
    *(_QWORD *)(a1 + 72) = 0;
  }
  *(_QWORD *)(a1 + 96) = a6;
  *(_BYTE *)(a1 + 104) = a8;
  *(_QWORD *)(a1 + 88) = a5;
  *(_WORD *)(a1 + 80) &= 0xF000u;
  *(_WORD *)(a1 + 26) = 0;
  return 0;
}
