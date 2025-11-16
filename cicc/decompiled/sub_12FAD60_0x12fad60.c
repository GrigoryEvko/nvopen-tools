// Function: sub_12FAD60
// Address: 0x12fad60
//
__int64 __fastcall sub_12FAD60(unsigned __int64 a1, __int64 a2, unsigned __int64 a3, __int64 a4)
{
  unsigned __int64 v5; // r9
  __int64 v6; // rax
  __int64 v8; // rcx
  __int64 v9; // r15
  __int64 v10; // r13
  unsigned __int8 v11; // bl
  unsigned __int64 v12; // r12
  __int64 v14; // rax
  __int64 v15; // rax
  __int64 v16; // r13
  unsigned __int64 v17; // r10
  __int64 v18; // r9
  __int64 v19; // rcx
  __int64 v20; // r8
  __int64 v21; // r10
  __int64 v22; // rsi
  __int64 v23; // rdi
  unsigned __int128 v24; // rax
  unsigned __int128 v25; // rax
  unsigned __int64 v26; // r8
  signed __int128 v27; // kr00_16
  __int64 v28; // rdx
  __int64 v29; // rsi
  __int64 v30; // rcx
  unsigned __int64 v31; // r9
  __int64 v32; // [rsp+8h] [rbp-68h]
  __int64 v33; // [rsp+10h] [rbp-60h]
  unsigned __int128 v34; // [rsp+10h] [rbp-60h]
  __int64 v35; // [rsp+10h] [rbp-60h]
  __int64 v36; // [rsp+20h] [rbp-50h] BYREF
  unsigned __int64 v37; // [rsp+28h] [rbp-48h]
  __int64 v38; // [rsp+30h] [rbp-40h]

  v5 = a3;
  v6 = HIWORD(a2) & 0x7FFF;
  v8 = a2 & 0xFFFFFFFFFFFFLL;
  v9 = a4 & 0xFFFFFFFFFFFFLL;
  v10 = HIWORD(a4) & 0x7FFF;
  v11 = (a4 < 0) ^ (a2 < 0);
  if ( v6 == 0x7FFF )
  {
    if ( a1 | v8 )
      return sub_12FBB40(a2, a1, a4, a3, a4, a3);
    v14 = a3 | v9;
    if ( v10 == 0x7FFF )
    {
      if ( v14 )
        return sub_12FBB40(a2, a1, a4, a3, a4, a3);
    }
    v15 = v10 | v14;
    if ( !v15 )
      goto LABEL_25;
    return 0;
  }
  if ( v10 == 0x7FFF )
  {
    if ( a3 | v9 )
      return sub_12FBB40(a2, a1, a4, a3, a4, a3);
    v15 = a1 | v8 | v6;
    if ( !v15 )
    {
LABEL_25:
      v35 = v15;
      sub_12F9B70(16);
      return v35;
    }
    return 0;
  }
  v12 = a1;
  if ( !v6 )
  {
    if ( !(a1 | v8) )
      return 0;
    sub_12FC3F0(&v36, v8, a1);
    v6 = v36;
    v12 = v37;
    v8 = v38;
    v5 = a3;
  }
  if ( !v10 )
  {
    v32 = v6;
    v33 = v8;
    if ( !(a3 | v9) )
      return 0;
    sub_12FC3F0(&v36, v9, a3);
    v10 = v36;
    v5 = v37;
    v9 = v38;
    v6 = v32;
    v8 = v33;
  }
  v16 = v6 + v10;
  v17 = v5;
  v18 = v5 << 16;
  v19 = v8 | 0x1000000000000LL;
  v20 = 0;
  v21 = (v9 << 16) | HIWORD(v17);
  v22 = v18 * v12;
  v23 = ((unsigned __int64)v18 * (unsigned __int128)v12) >> 64;
  v24 = (unsigned __int64)v21 * (unsigned __int128)v12;
  v34 = v24 + (unsigned __int64)v19 * (unsigned __int128)(unsigned __int64)v18;
  v31 = __CFADD__(
          __CFADD__((_QWORD)v24, v19 * v18),
          ((unsigned __int64)v19 * (unsigned __int128)(unsigned __int64)v18) >> 64)
      | (unsigned __int8)__CFADD__(*((_QWORD *)&v24 + 1), *((_QWORD *)&v34 + 1));
  if ( v31 )
  {
    v31 = 0;
    v20 = 1;
  }
  v25 = __PAIR128__(v20, v31 | *((_QWORD *)&v34 + 1))
      + __CFADD__((_QWORD)v34, v23)
      + (unsigned __int64)v19 * (unsigned __int128)(unsigned __int64)v21;
  v26 = (v22 != 0) | (unsigned __int64)(v34 + v23);
  v27 = __PAIR128__(*((_QWORD *)&v25 + 1) + v19, v25) + v12;
  if ( *((_QWORD *)&v27 + 1) > 0x1FFFFFFFFFFFFuLL )
  {
    v29 = v16 - 0x3FFF;
    v28 = *((_QWORD *)&v27 + 1) >> 1;
    v30 = v27 >> 1;
    v26 = ((_QWORD)v27 << 63) | (v26 != 0);
  }
  else
  {
    v28 = (__PAIR128__(*((_QWORD *)&v25 + 1) + v19, v25) + v12) >> 64;
    v29 = v16 - 0x4000;
    v30 = v27;
  }
  return sub_12FC4A0(v11, v29, v28, v30, v26);
}
