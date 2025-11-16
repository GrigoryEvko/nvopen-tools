// Function: sub_30E01C0
// Address: 0x30e01c0
//
__int64 __fastcall sub_30E01C0(
        __int64 a1,
        __int64 a2,
        __int64 a3,
        __int64 a4,
        __int64 a5,
        __int64 a6,
        __int64 a7,
        __int64 a8,
        __int64 a9,
        __int64 a10)
{
  __int64 v11; // rsi
  int v12; // eax
  __int64 v13; // rdi
  __int64 v14; // rax
  int v15; // ecx
  __int64 v17; // [rsp+8h] [rbp-3D8h]
  int v18[24]; // [rsp+10h] [rbp-3D0h] BYREF
  _QWORD v19[9]; // [rsp+70h] [rbp-370h] BYREF
  __int64 v20; // [rsp+B8h] [rbp-328h]
  __int64 v21; // [rsp+C0h] [rbp-320h]
  unsigned __int8 *v22; // [rsp+D0h] [rbp-310h]
  char v23; // [rsp+2F8h] [rbp-E8h]
  int v24; // [rsp+300h] [rbp-E0h]
  int v25; // [rsp+304h] [rbp-DCh]
  __int64 v26; // [rsp+308h] [rbp-D8h]
  int v27; // [rsp+330h] [rbp-B0h]
  int v28; // [rsp+33Ch] [rbp-A4h]

  v11 = *(_QWORD *)(a1 - 32);
  memset(v18, 0, 84);
  *(_QWORD *)&v18[15] = 0x1010001010101LL;
  if ( v11 )
  {
    if ( *(_BYTE *)v11 )
    {
      v11 = 0;
    }
    else if ( *(_QWORD *)(v11 + 24) != *(_QWORD *)(a1 + 80) )
    {
      v11 = 0;
    }
  }
  sub_30D4900((__int64)v19, v11, a1, v18, a2, a9, a3, a4, a5, a6, a7, a8, a10, 1, 1);
  sub_30D2590((__int64)v19, (__int64)v22, v20);
  v27 += v25 + v24;
  v12 = sub_30D4FE0((__int64 *)v19[1], v22, v21);
  v13 = v20;
  v14 = v28 + (__int64)-v12;
  if ( v14 > 0x7FFFFFFF )
    v14 = 0x7FFFFFFF;
  if ( v14 < (__int64)0xFFFFFFFF80000000LL )
    LODWORD(v14) = 0x80000000;
  v28 = v14;
  v15 = v14;
  if ( ((*(_WORD *)(v20 + 2) >> 4) & 0x3FF) == 9 )
  {
    v15 = v14 + 2000;
    v28 = v14 + 2000;
  }
  if ( v27 <= v15 && !v23 )
    goto LABEL_13;
  if ( !*(_BYTE *)(v26 + 66) )
  {
    if ( sub_B2DCC0(v20) )
      goto LABEL_13;
    v13 = v20;
  }
  if ( v13 + 72 == (*(_QWORD *)(v13 + 72) & 0xFFFFFFFFFFFFFFF8LL) || !sub_30DC7E0(v19) )
  {
    BYTE4(v17) = 1;
    LODWORD(v17) = v28;
    goto LABEL_14;
  }
LABEL_13:
  BYTE4(v17) = 0;
LABEL_14:
  sub_30D30A0((__int64)v19);
  return v17;
}
