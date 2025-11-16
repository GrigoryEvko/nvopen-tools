// Function: sub_12FB050
// Address: 0x12fb050
//
__int64 __fastcall sub_12FB050(unsigned __int64 a1, __int64 a2, unsigned __int64 a3, __int64 a4)
{
  __int64 v6; // r9
  __int64 v7; // rcx
  __int64 v8; // rax
  __int64 v9; // r13
  unsigned __int8 v10; // bl
  unsigned __int64 v11; // r15
  unsigned __int64 v12; // r12
  __int64 v14; // r13
  __int64 v15; // rdi
  unsigned __int64 v16; // rcx
  unsigned __int64 v17; // r9
  __int64 v18; // rsi
  unsigned __int64 *v19; // r10
  __int64 v20; // r11
  unsigned __int64 v21; // r8
  unsigned __int64 v22; // kr00_8
  __int64 v23; // rdi
  unsigned __int64 v24; // kr20_8
  unsigned __int64 v25; // rdx
  unsigned __int128 v26; // kr40_16
  __int64 v27; // [rsp+8h] [rbp-88h]
  __int64 v28; // [rsp+10h] [rbp-80h]
  __int64 v29; // [rsp+20h] [rbp-70h] BYREF
  unsigned __int64 v30; // [rsp+28h] [rbp-68h] BYREF
  __int64 v31; // [rsp+30h] [rbp-60h]
  __int64 v32; // [rsp+40h] [rbp-50h] BYREF
  unsigned __int64 v33; // [rsp+48h] [rbp-48h]
  __int64 v34; // [rsp+50h] [rbp-40h]

  v6 = a2 & 0xFFFFFFFFFFFFLL;
  v7 = a4 & 0xFFFFFFFFFFFFLL;
  v8 = HIWORD(a4) & 0x7FFF;
  v9 = HIWORD(a2) & 0x7FFF;
  v10 = (a4 < 0) ^ (a2 < 0);
  if ( v9 == 0x7FFF )
  {
    v6 |= a1;
    if ( !v6 )
    {
      if ( v8 != 0x7FFF )
        return 0;
      if ( !(a3 | v7) )
        goto LABEL_6;
    }
    return sub_12FBB40(a2, a1, a4, a3, a3, v6);
  }
  if ( v8 == 0x7FFF )
  {
    if ( !(a3 | v7) )
      return 0;
    return sub_12FBB40(a2, a1, a4, a3, a3, v6);
  }
  v11 = a1;
  v12 = a3;
  if ( !v8 )
  {
    if ( !(v7 | a3) )
    {
      if ( !(v9 | a1 | v6) )
      {
LABEL_6:
        sub_12F9B70(16);
        return 0;
      }
      sub_12F9B70(8);
      return 0;
    }
    sub_12FC3F0(&v29, v7, a3);
    v8 = v29;
    v12 = v30;
    v7 = v31;
    v6 = a2 & 0xFFFFFFFFFFFFLL;
  }
  if ( !v9 )
  {
    v27 = v8;
    v28 = v7;
    if ( a1 | v6 )
    {
      sub_12FC3F0(&v29, v6, a1);
      v9 = v29;
      v11 = v30;
      v6 = v31;
      v8 = v27;
      v7 = v28;
      goto LABEL_17;
    }
    return 0;
  }
LABEL_17:
  v14 = v9 - v8;
  v15 = v6 | 0x1000000000000LL;
  v16 = v7 | 0x1000000000000LL;
  if ( (v6 | 0x1000000000000uLL) < v16 || v11 < v12 && v15 == v16 )
  {
    v17 = 2 * v11;
    v18 = v14 + 16381;
    v15 = (v11 > 2 * v11) + 2 * v15;
  }
  else
  {
    v18 = v14 + 16382;
    v17 = v11;
  }
  v19 = (unsigned __int64 *)&v32;
  v20 = (unsigned int)(0x7FFFFFFFFFFFFFFFuLL / (unsigned int)(v16 >> 17));
  v21 = (v20 * (unsigned __int64)(unsigned int)((unsigned __int64)v15 >> 19) + 0x80000000) >> 32;
  do
  {
    v22 = (v17 << 29) - v12 * v21;
    v15 = (__PAIR128__(((v17 >> 35) | (v15 << 29)) - ((__PAIR128__(v16, v12) * v21) >> 64), v17 << 29) - v12 * v21) >> 64;
    v17 = v22;
    if ( v15 < 0 )
    {
      --v21;
      v15 = (__PAIR128__(v16, v12) + __PAIR128__(v15, v22)) >> 64;
      v17 = v12 + v22;
    }
    v19[2] = v21;
    --v19;
    v21 = (v20 * (unsigned __int64)(unsigned int)((unsigned __int64)v15 >> 19) + 0x80000000) >> 32;
  }
  while ( &v30 != v19 );
  if ( (((_BYTE)v21 + 1) & 6) == 0 )
  {
    v24 = (v17 << 29) - v12 * v21;
    v23 = (__PAIR128__(((v17 >> 35) | (v15 << 29)) - ((__PAIR128__(v16, v12) * v21) >> 64), v17 << 29) - v12 * v21) >> 64;
    if ( v23 < 0 )
    {
      --v21;
      v23 = (__PAIR128__(v16, v12) + __PAIR128__(v23, v24)) >> 64;
      v25 = v12 + v24;
    }
    else if ( __PAIR128__(v16, v12) > __PAIR128__(v23, v24) )
    {
      v25 = (v17 << 29) - v12 * v21;
    }
    else
    {
      ++v21;
      v25 = v24 - v12;
      v23 = ((__PAIR128__(v23, v24) - v12) >> 64) - v16;
    }
    if ( v25 | v23 )
      v21 |= 1u;
  }
  v26 = __PAIR128__((v34 << 19) + (v33 >> 10), v33 << 54) + (v21 >> 4) + (v32 << 25);
  return sub_12FC4A0(v10, v18, *((_QWORD *)&v26 + 1), v26, v21 << 60);
}
