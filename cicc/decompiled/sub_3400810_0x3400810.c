// Function: sub_3400810
// Address: 0x3400810
//
unsigned __int8 *__fastcall sub_3400810(
        _QWORD *a1,
        __int64 a2,
        __int64 a3,
        __int64 a4,
        __int64 a5,
        __int64 a6,
        __m128i a7,
        __int128 a8,
        unsigned __int16 a9,
        __int64 a10)
{
  __int64 v15; // rax
  __int64 v16; // rsi
  __int16 v17; // dx
  __int64 v18; // rcx
  unsigned __int16 v19; // ax
  unsigned __int8 *result; // rax
  __int64 v21; // rdx
  __int64 v22; // rcx
  __int64 v23; // r8
  __int64 v24; // rax
  __int64 v25; // rdi
  __int64 v26; // rdx
  __int64 v27; // r8
  __int64 v28; // rax
  __int64 v29; // rax
  __int64 v30; // rdx
  unsigned __int64 v31; // rax
  __int128 v32; // rax
  __int64 v33; // r9
  bool v34; // al
  __int64 v35; // rcx
  __int16 v36; // ax
  __int64 v37; // r9
  __int64 v38; // rdx
  __int128 v39; // [rsp-50h] [rbp-F0h]
  __int128 v40; // [rsp-30h] [rbp-D0h]
  unsigned int v41; // [rsp+Ch] [rbp-94h]
  __int64 v42; // [rsp+10h] [rbp-90h]
  int v43; // [rsp+10h] [rbp-90h]
  unsigned int v44; // [rsp+10h] [rbp-90h]
  int v45; // [rsp+10h] [rbp-90h]
  unsigned __int8 *v47; // [rsp+20h] [rbp-80h]
  unsigned int v48; // [rsp+30h] [rbp-70h] BYREF
  __int64 v49; // [rsp+38h] [rbp-68h]
  unsigned __int64 v50; // [rsp+40h] [rbp-60h] BYREF
  __int64 v51; // [rsp+48h] [rbp-58h]
  __int64 v52; // [rsp+50h] [rbp-50h]
  __int64 v53; // [rsp+58h] [rbp-48h]
  __int64 v54; // [rsp+60h] [rbp-40h] BYREF
  __int64 v55; // [rsp+68h] [rbp-38h]

  v15 = *(_QWORD *)(a2 + 48) + 16LL * (unsigned int)a3;
  v16 = a10;
  v17 = *(_WORD *)v15;
  v18 = *(_QWORD *)(v15 + 8);
  v19 = a9;
  LOWORD(v48) = v17;
  v49 = v18;
  if ( v17 == a9 && (a9 || v18 == a10) )
    return (unsigned __int8 *)a2;
  if ( a9 )
  {
    if ( (unsigned __int16)(a9 - 17) > 0xD3u )
    {
      LOWORD(v54) = a9;
      v55 = a10;
      goto LABEL_24;
    }
    v19 = word_4456580[a9 - 1];
    v38 = 0;
  }
  else
  {
    v42 = a10;
    v16 = a10;
    if ( !sub_30070B0((__int64)&a9) )
    {
      v55 = v42;
      LOWORD(v54) = 0;
      goto LABEL_9;
    }
    v19 = sub_3009970((__int64)&a9, v16, v21, v22, v23);
  }
  LOWORD(v54) = v19;
  v55 = v38;
  if ( !v19 )
  {
LABEL_9:
    v24 = sub_3007260((__int64)&v54);
    v25 = v26;
    LODWORD(v26) = (unsigned __int16)v48;
    v52 = v24;
    LODWORD(v27) = v24;
    v53 = v25;
    if ( (_WORD)v48 )
      goto LABEL_10;
LABEL_27:
    v41 = v26;
    v44 = v27;
    v34 = sub_30070B0((__int64)&v48);
    LODWORD(v27) = v44;
    LOWORD(v26) = v41;
    if ( v34 )
    {
      v36 = sub_3009970((__int64)&v48, v16, v41, v35, v44);
      LODWORD(v27) = v44;
      v37 = v26;
      LOWORD(v26) = v36;
      v28 = v37;
LABEL_12:
      LOWORD(v50) = v26;
      v51 = v28;
      if ( !(_WORD)v26 )
        goto LABEL_13;
      goto LABEL_34;
    }
LABEL_11:
    v28 = v49;
    goto LABEL_12;
  }
LABEL_24:
  if ( v19 == 1 || (unsigned __int16)(v19 - 504) <= 7u )
    goto LABEL_40;
  v27 = *(_QWORD *)&byte_444C4A0[16 * v19 - 16];
  LODWORD(v26) = (unsigned __int16)v48;
  if ( !(_WORD)v48 )
    goto LABEL_27;
LABEL_10:
  if ( (unsigned __int16)(v26 - 17) > 0xD3u )
    goto LABEL_11;
  LOWORD(v26) = word_4456580[(int)v26 - 1];
  v51 = 0;
  LOWORD(v50) = v26;
  if ( (_WORD)v26 )
  {
LABEL_34:
    if ( (_WORD)v26 != 1 && (unsigned __int16)(v26 - 504) > 7u )
    {
      LODWORD(v51) = *(_QWORD *)&byte_444C4A0[16 * (unsigned __int16)v26 - 16];
      if ( (unsigned int)v51 <= 0x40 )
        goto LABEL_14;
LABEL_37:
      v45 = v27;
      sub_C43690((__int64)&v50, 0, 0);
      LODWORD(v27) = v45;
      goto LABEL_15;
    }
LABEL_40:
    BUG();
  }
LABEL_13:
  v43 = v27;
  v29 = sub_3007260((__int64)&v50);
  LODWORD(v27) = v43;
  v54 = v29;
  v55 = v30;
  LODWORD(v51) = v29;
  if ( (unsigned int)v29 > 0x40 )
    goto LABEL_37;
LABEL_14:
  v50 = 0;
LABEL_15:
  if ( (_DWORD)v27 )
  {
    if ( (unsigned int)v27 > 0x40 )
    {
      sub_C43C90(&v50, 0, v27);
    }
    else
    {
      v31 = 0xFFFFFFFFFFFFFFFFLL >> (64 - (unsigned __int8)v27);
      if ( (unsigned int)v51 > 0x40 )
        *(_QWORD *)v50 |= v31;
      else
        v50 |= v31;
    }
  }
  *(_QWORD *)&v32 = sub_34007B0((__int64)a1, (__int64)&v50, a6, v48, v49, 0, a7, 0);
  *((_QWORD *)&v40 + 1) = a5;
  *(_QWORD *)&v40 = a4;
  *((_QWORD *)&v39 + 1) = a3;
  *(_QWORD *)&v39 = a2;
  result = sub_33FC130(a1, 396, a6, v48, v49, v33, v39, v32, v40, a8);
  if ( (unsigned int)v51 > 0x40 )
  {
    if ( v50 )
    {
      v47 = result;
      j_j___libc_free_0_0(v50);
      return v47;
    }
  }
  return result;
}
