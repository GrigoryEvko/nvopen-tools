// Function: sub_1349DA0
// Address: 0x1349da0
//
__int64 __fastcall sub_1349DA0(_QWORD *a1, __int64 a2, unsigned __int64 a3)
{
  _QWORD *v3; // r15
  unsigned __int64 v4; // r12
  unsigned __int64 v6; // rsi
  unsigned __int64 v7; // r13
  unsigned __int64 v8; // r8
  unsigned __int64 v9; // r14
  __int64 v10; // rdx
  __int64 v11; // r9
  unsigned __int64 *v12; // r10
  unsigned __int64 v13; // rdi
  unsigned __int64 v14; // rdx
  unsigned __int64 v15; // rax
  unsigned __int64 v17; // rsi
  __int64 v18; // rax
  unsigned __int64 v19; // rsi
  unsigned __int64 v20; // rcx
  __int64 result; // rax
  bool v22; // zf
  __int64 v23; // rsi
  unsigned __int64 v24; // rsi
  unsigned __int64 v25; // r11
  __int64 v26; // rsi
  unsigned __int64 *v27; // [rsp+8h] [rbp-58h]
  char v28; // [rsp+20h] [rbp-40h]
  unsigned __int64 v29; // [rsp+28h] [rbp-38h]

  v3 = a1 + 14;
  v4 = a3 >> 12;
  v6 = a2 - *a1;
  v7 = a1[12];
  v8 = v6 >> 12;
  v9 = v6 >> 18;
  v10 = v6 >> 18;
  v11 = (v6 >> 12) & 0x3F;
  v12 = &a1[v10 + 14];
  v13 = *v12;
  if ( v4 + v11 <= 0x40 )
  {
    *v12 = v13 & ~(0xFFFFFFFFFFFFFFFFLL >> (64 - (unsigned __int8)v4) << v11);
    goto LABEL_3;
  }
  v25 = v4 + v11 - 64;
  *v12 = v13 & ~(0xFFFFFFFFFFFFFFFFLL >> v11 << v11);
  if ( v25 > 0x40 )
  {
    v27 = v12;
    v28 = v4 + v11 - 64;
    v29 = (v4 + v11 - 129) >> 6;
    memset(&v3[v10 + 1], 0, 8 * v29 + 8);
    v8 = v6 >> 12;
    v11 = (v6 >> 12) & 0x3F;
    v26 = v9 + v29 + 2;
    v12 = v27;
    LOBYTE(v25) = v28 - ((_BYTE)v29 << 6) - 64;
LABEL_22:
    v3[v26] &= ~(0xFFFFFFFFFFFFFFFFLL >> (64 - (unsigned __int8)v25));
    goto LABEL_3;
  }
  v26 = v9 + 1;
  if ( v4 + v11 != 64 )
    goto LABEL_22;
LABEL_3:
  v14 = *v12 & ((2LL << v11) - 1);
  if ( v14 )
  {
LABEL_8:
    _BitScanReverse64(&v14, v14);
    v14 = (int)v14 + (v9 << 6) + 1;
    goto LABEL_9;
  }
  v15 = v9 - 1;
  if ( v9 )
  {
    do
    {
      v14 = a1[v15 + 14];
      v9 = v15;
      if ( v14 )
        goto LABEL_8;
    }
    while ( v15-- != 0 );
  }
LABEL_9:
  v17 = v8 + v4 - 1;
  v18 = -1LL << (v17 & 0x3F);
  v19 = v17 >> 6;
  v20 = v3[v19] & v18;
  if ( !v20 )
  {
    result = v19 + 1;
    if ( v19 == 7 )
    {
LABEL_23:
      v23 = 512;
      goto LABEL_17;
    }
    while ( 1 )
    {
      v20 = a1[result + 14];
      v19 = result;
      if ( v20 )
        break;
      if ( ++result == 8 )
        goto LABEL_23;
    }
  }
  v22 = !_BitScanForward64(&v20, v20);
  result = -1;
  if ( v22 )
    LODWORD(v20) = -1;
  v23 = (int)v20 + (v19 << 6);
LABEL_17:
  v24 = v23 - v14;
  if ( v24 > v7 )
    a1[12] = v24;
  a1[13] -= v4;
  return result;
}
