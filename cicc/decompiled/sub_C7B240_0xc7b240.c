// Function: sub_C7B240
// Address: 0xc7b240
//
__int64 __fastcall sub_C7B240(__int64 a1, __int64 a2, __int64 a3)
{
  unsigned int v4; // ebx
  int v5; // r14d
  int v6; // eax
  unsigned int v7; // eax
  unsigned __int64 v8; // rax
  unsigned int v9; // edx
  unsigned int v10; // esi
  char v12; // cl
  unsigned __int64 v13; // rbx
  unsigned int v14; // eax
  unsigned int v15; // edx
  unsigned __int64 v16; // rax
  __int64 v17; // [rsp+0h] [rbp-70h]
  unsigned int v18; // [rsp+Ch] [rbp-64h]
  unsigned __int64 v19; // [rsp+10h] [rbp-60h] BYREF
  unsigned int v20; // [rsp+18h] [rbp-58h]
  __int64 v21; // [rsp+20h] [rbp-50h] BYREF
  unsigned int v22; // [rsp+28h] [rbp-48h]
  __int64 v23; // [rsp+30h] [rbp-40h] BYREF
  unsigned int v24; // [rsp+38h] [rbp-38h]

  sub_C7AF00(a1, a2, a3);
  v4 = *(_DWORD *)(a3 + 8);
  if ( v4 > 0x40 )
    v5 = sub_C44630(a3);
  else
    v5 = sub_39FAC40(*(_QWORD *)a3);
  v18 = *(_DWORD *)(a3 + 24);
  if ( v18 > 0x40 )
  {
    v6 = sub_C44630(a3 + 16);
    if ( v6 + v5 == v4 && v6 == 1 )
    {
      v22 = v18;
      sub_C43780((__int64)&v21, (const void **)(a3 + 16));
      goto LABEL_31;
    }
  }
  else
  {
    v17 = *(_QWORD *)(a3 + 16);
    if ( (unsigned int)sub_39FAC40(v17) + v5 == v4 && v17 && (v17 & (v17 - 1)) == 0 )
    {
      v21 = v17;
      v22 = v18;
LABEL_31:
      sub_C46F20((__int64)&v21, 1u);
      v15 = v22;
      v22 = 0;
      v24 = v15;
      v23 = v21;
      if ( v15 > 0x40 )
      {
        sub_C43D10((__int64)&v23);
        v20 = v24;
        v19 = v23;
        if ( v22 > 0x40 && v21 )
          j_j___libc_free_0_0(v21);
      }
      else
      {
        v20 = v15;
        v16 = (0xFFFFFFFFFFFFFFFFLL >> -(char)v15) & ~v21;
        if ( !v15 )
          v16 = 0;
        v19 = v16;
      }
      if ( *(_DWORD *)(a1 + 8) > 0x40u )
        sub_C43BD0((_QWORD *)a1, (__int64 *)&v19);
      else
        *(_QWORD *)a1 |= v19;
      if ( v20 > 0x40 && v19 )
        j_j___libc_free_0_0(v19);
      return a1;
    }
  }
  if ( v4 > 0x40 )
  {
    v4 = sub_C44500(a3);
    goto LABEL_12;
  }
  if ( !v4 || (v12 = 64 - v4, v4 = 64, *(_QWORD *)a3 << v12 == -1) )
  {
LABEL_12:
    v7 = *(_DWORD *)(a2 + 8);
    if ( v7 <= 0x40 )
      goto LABEL_13;
LABEL_25:
    v14 = sub_C44500(a2);
    if ( v4 < v14 )
      v4 = v14;
    goto LABEL_17;
  }
  _BitScanReverse64(&v13, ~(*(_QWORD *)a3 << v12));
  v7 = *(_DWORD *)(a2 + 8);
  v4 = v13 ^ 0x3F;
  if ( v7 > 0x40 )
    goto LABEL_25;
LABEL_13:
  if ( v7 )
  {
    v8 = ~(*(_QWORD *)a2 << (64 - (unsigned __int8)v7));
    if ( v8 )
    {
      _BitScanReverse64(&v8, v8);
      LODWORD(v8) = v8 ^ 0x3F;
      if ( v4 < (unsigned int)v8 )
        v4 = v8;
    }
    else if ( v4 < 0x40 )
    {
      v4 = 64;
    }
  }
LABEL_17:
  v9 = *(_DWORD *)(a1 + 8);
  v10 = v9 - v4;
  if ( v9 != v9 - v4 )
  {
    if ( v10 > 0x3F || v9 > 0x40 )
      sub_C43C90((_QWORD *)a1, v10, v9);
    else
      *(_QWORD *)a1 |= 0xFFFFFFFFFFFFFFFFLL >> (64 - (unsigned __int8)v4) << v10;
  }
  return a1;
}
