// Function: sub_30DFEC0
// Address: 0x30dfec0
//
__int64 __fastcall sub_30DFEC0(__int64 a1, __int64 a2, __int64 a3)
{
  __int64 v4; // r9
  __int64 v5; // r8
  int v6; // eax
  __int64 v7; // rdi
  __int64 v8; // rax
  int v9; // ecx
  int v10; // eax
  int v12[24]; // [rsp+8h] [rbp-3C0h] BYREF
  _QWORD v13[9]; // [rsp+68h] [rbp-360h] BYREF
  __int64 v14; // [rsp+B0h] [rbp-318h]
  __int64 v15; // [rsp+B8h] [rbp-310h]
  unsigned __int8 *v16; // [rsp+C8h] [rbp-300h]
  char v17; // [rsp+2F0h] [rbp-D8h]
  int v18; // [rsp+2F8h] [rbp-D0h]
  int v19; // [rsp+2FCh] [rbp-CCh]
  __int64 v20; // [rsp+300h] [rbp-C8h]
  int v21; // [rsp+328h] [rbp-A0h]
  int v22; // [rsp+334h] [rbp-94h]

  v4 = *(_QWORD *)(a1 + 64);
  v5 = *(_QWORD *)(a1 + 8);
  memset(v12, 0, 84);
  v12[0] = 100;
  *(_QWORD *)&v12[15] = 0x1010001010101LL;
  sub_30D4900(
    (__int64)v13,
    a2,
    a3,
    v12,
    v5,
    v4,
    *(_QWORD *)(a1 + 16),
    *(_QWORD *)(a1 + 24),
    *(_QWORD *)(a1 + 32),
    *(_QWORD *)(a1 + 40),
    *(_QWORD *)(a1 + 48),
    *(_QWORD *)(a1 + 56),
    *(_QWORD *)(a1 + 88),
    0,
    1);
  sub_30D2590((__int64)v13, (__int64)v16, v14);
  v21 += v19 + v18;
  v6 = sub_30D4FE0((__int64 *)v13[1], v16, v15);
  v7 = v14;
  v8 = v22 + (__int64)-v6;
  if ( v8 > 0x7FFFFFFF )
    v8 = 0x7FFFFFFF;
  if ( v8 < (__int64)0xFFFFFFFF80000000LL )
    LODWORD(v8) = 0x80000000;
  v22 = v8;
  v9 = v8;
  if ( ((*(_WORD *)(v14 + 2) >> 4) & 0x3FF) == 9 )
  {
    v9 = v8 + 2000;
    v22 = v8 + 2000;
  }
  if ( v21 > v9 || v17 )
  {
    if ( !*(_BYTE *)(v20 + 66) )
    {
      if ( sub_B2DCC0(v14) )
        return sub_30D30A0((__int64)v13);
      v7 = v14;
    }
    if ( v7 + 72 == (*(_QWORD *)(v7 + 72) & 0xFFFFFFFFFFFFFFF8LL) || !sub_30DC7E0(v13) )
    {
      v10 = v22;
      ++*(_DWORD *)(a1 + 736);
      *(_DWORD *)(a1 + 740) += v10;
    }
  }
  return sub_30D30A0((__int64)v13);
}
