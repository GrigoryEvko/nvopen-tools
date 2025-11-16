// Function: sub_21CD550
// Address: 0x21cd550
//
__int64 *__fastcall sub_21CD550(__int64 a1, __int64 a2, double a3, double a4, __m128i a5)
{
  __int64 *v5; // r13
  char *v6; // rax
  char v7; // bl
  __int64 v9; // rsi
  unsigned int v10; // ecx
  __int64 v11; // rax
  __m128i v12; // xmm0
  __m128i v13; // xmm1
  __int64 v14; // r14
  __int64 v15; // r13
  int v16; // eax
  int v17; // eax
  __int64 v19; // rsi
  unsigned int v20; // r13d
  __int64 *v21; // r15
  unsigned int v22; // ebx
  __int64 v23; // r9
  int v24; // eax
  __int64 v25; // rax
  unsigned int v26; // ebx
  unsigned int v27; // ecx
  int v28; // r11d
  int v29; // eax
  bool v30; // al
  __int64 v31; // rax
  __int64 *v32; // rdi
  unsigned __int64 v33; // rdx
  unsigned __int64 v34; // r15
  __int64 v35; // r14
  __int128 v36; // rax
  __int64 v37; // rdx
  unsigned int v38; // r15d
  __int64 v39; // rdi
  unsigned __int64 v40; // rsi
  unsigned int v41; // edx
  __int64 v42; // rax
  int v43; // eax
  unsigned __int64 v44; // rsi
  unsigned __int64 v45; // rax
  unsigned int v46; // edx
  unsigned int v47; // ecx
  unsigned int v48; // r13d
  int v49; // eax
  __int64 v50; // rdx
  __int64 v51; // rax
  __int64 v52; // rdx
  unsigned __int64 v53; // rdx
  int v54; // eax
  unsigned __int64 v55; // rax
  int v56; // eax
  __int128 v57; // [rsp-10h] [rbp-100h]
  __int64 v58; // [rsp+0h] [rbp-F0h]
  __int64 v59; // [rsp+10h] [rbp-E0h]
  int v60; // [rsp+10h] [rbp-E0h]
  int v61; // [rsp+10h] [rbp-E0h]
  unsigned int v62; // [rsp+1Ch] [rbp-D4h]
  int v63; // [rsp+1Ch] [rbp-D4h]
  int v64; // [rsp+1Ch] [rbp-D4h]
  int v65; // [rsp+1Ch] [rbp-D4h]
  __int64 v66; // [rsp+20h] [rbp-D0h]
  unsigned __int64 v67; // [rsp+28h] [rbp-C8h]
  __m128i v68; // [rsp+30h] [rbp-C0h] BYREF
  __int64 v69; // [rsp+40h] [rbp-B0h]
  __int64 v70; // [rsp+48h] [rbp-A8h]
  __m128i v71; // [rsp+50h] [rbp-A0h]
  __m128i v72; // [rsp+60h] [rbp-90h]
  unsigned int v73; // [rsp+70h] [rbp-80h] BYREF
  const void **v74; // [rsp+78h] [rbp-78h]
  __int64 v75; // [rsp+80h] [rbp-70h] BYREF
  int v76; // [rsp+88h] [rbp-68h]
  __int64 *v77; // [rsp+90h] [rbp-60h] BYREF
  unsigned int v78; // [rsp+98h] [rbp-58h]
  unsigned __int64 v79; // [rsp+A0h] [rbp-50h] BYREF
  unsigned int v80; // [rsp+A8h] [rbp-48h]
  unsigned __int64 v81; // [rsp+B0h] [rbp-40h] BYREF
  unsigned int v82; // [rsp+B8h] [rbp-38h]

  v5 = 0;
  v6 = *(char **)(a1 + 40);
  v7 = *v6;
  v74 = (const void **)*((_QWORD *)v6 + 1);
  LOBYTE(v73) = v7;
  if ( (unsigned __int8)(v7 - 5) > 1u )
    return v5;
  v9 = *(_QWORD *)(a1 + 72);
  v75 = v9;
  if ( v9 )
  {
    sub_1623A60((__int64)&v75, v9, 2);
    v7 = v73;
    v76 = *(_DWORD *)(a1 + 64);
    if ( !(_BYTE)v73 )
    {
      v68.m128i_i64[0] = (__int64)&v73;
      v62 = sub_1F58D40((__int64)&v73);
      goto LABEL_5;
    }
  }
  else
  {
    v76 = *(_DWORD *)(a1 + 64);
  }
  v68.m128i_i64[0] = (__int64)&v73;
  v62 = sub_1F3E310(&v73);
LABEL_5:
  v11 = *(_QWORD *)(a1 + 32);
  v12 = _mm_loadu_si128((const __m128i *)v11);
  v13 = _mm_loadu_si128((const __m128i *)(v11 + 40));
  v14 = *(_QWORD *)v11;
  v15 = *(_QWORD *)(v11 + 40);
  v16 = *(unsigned __int16 *)(a1 + 24);
  v67 = v12.m128i_u64[1];
  v68 = v13;
  if ( v16 != 54 )
  {
    if ( v16 != 122 )
      goto LABEL_26;
    v17 = *(unsigned __int16 *)(v15 + 24);
    if ( v17 != 10 && v17 != 32 )
      goto LABEL_9;
    v19 = *(_QWORD *)(v15 + 88);
    v20 = *(_DWORD *)(v19 + 32);
    v78 = v20;
    if ( v20 > 0x40 )
    {
      sub_16A4FD0((__int64)&v77, (const void **)(v19 + 24));
      v20 = v78;
      v21 = v77;
      v7 = v73;
    }
    else
    {
      v21 = *(__int64 **)(v19 + 24);
      v77 = v21;
    }
    if ( v7 )
      v22 = sub_1F3E310(&v73);
    else
      v22 = sub_1F58D40((__int64)&v73);
    if ( v20 <= 0x40 )
    {
      v23 = (__int64)((_QWORD)v21 << (64 - (unsigned __int8)v20)) >> (64 - (unsigned __int8)v20);
      if ( v23 < 0 || v23 >= v22 )
        goto LABEL_9;
LABEL_57:
      v82 = v22;
      if ( v22 > 0x40 )
      {
        sub_16A4EF0((__int64)&v81, 1, 0);
        v80 = v82;
        if ( v82 > 0x40 )
        {
          sub_16A4FD0((__int64)&v79, (const void **)&v81);
LABEL_60:
          sub_16A7E20((__int64)&v79, (__int64)&v77);
          if ( v82 > 0x40 && v81 )
            j_j___libc_free_0_0(v81);
          v51 = sub_1D38970(
                  *(_QWORD *)(a2 + 16),
                  (__int64)&v79,
                  (__int64)&v75,
                  v73,
                  v74,
                  0,
                  v12,
                  *(double *)v13.m128i_i64,
                  a5,
                  0);
          v10 = 0;
          v69 = v51;
          v15 = v51;
          v68.m128i_i64[0] = v51;
          v70 = v52;
          v68.m128i_i64[1] = (unsigned int)v52 | v68.m128i_i64[1] & 0xFFFFFFFF00000000LL;
          if ( v80 > 0x40 && v79 )
            j_j___libc_free_0_0(v79);
          if ( v78 > 0x40 && v77 )
            j_j___libc_free_0_0(v77);
          goto LABEL_26;
        }
      }
      else
      {
        v80 = v22;
        v81 = (0xFFFFFFFFFFFFFFFFLL >> -(char)v22) & 1;
      }
      v79 = v81;
      goto LABEL_60;
    }
    v47 = v20 - 1;
    v48 = v20 + 1;
    if ( (v21[v47 >> 6] & (1LL << v47)) != 0 )
    {
      if ( v48 - (unsigned int)sub_16A5810((__int64)&v77) > 0x40 || *v21 < 0 )
        goto LABEL_48;
      v58 = *v21;
      v59 = v22;
      v49 = sub_16A5810((__int64)&v77);
      v50 = v58;
      if ( v48 - v49 > 0x40 )
        goto LABEL_57;
    }
    else
    {
      if ( v48 - (unsigned int)sub_16A57B0((__int64)&v77) > 0x40 )
        goto LABEL_48;
      if ( *v21 < 0 )
        goto LABEL_48;
      v59 = v22;
      if ( v48 - (unsigned int)sub_16A57B0((__int64)&v77) > 0x40 )
        goto LABEL_48;
      v50 = *v21;
    }
    if ( v59 > v50 )
      goto LABEL_57;
LABEL_48:
    v5 = 0;
    if ( v21 )
      j_j___libc_free_0_0(v21);
    goto LABEL_10;
  }
  v24 = *(unsigned __int16 *)(v14 + 24);
  if ( v24 == 32 || v24 == 10 )
  {
    a5 = _mm_load_si128(&v68);
    v10 = 0;
    v72 = a5;
    v71 = v12;
    v68.m128i_i64[0] = v12.m128i_i64[0];
    v67 = a5.m128i_u32[2] | v12.m128i_i64[1] & 0xFFFFFFFF00000000LL;
    v68.m128i_i64[1] = v12.m128i_u32[2] | v68.m128i_i64[1] & 0xFFFFFFFF00000000LL;
    v25 = v15;
    v15 = v14;
    v14 = v25;
  }
LABEL_26:
  v26 = v62 >> 1;
  if ( !(unsigned __int8)sub_21CB9A0(v14, v62 >> 1, &v79, v10) )
    goto LABEL_9;
  v28 = v79;
  if ( (_DWORD)v79 == 2 )
    goto LABEL_9;
  v29 = *(unsigned __int16 *)(v15 + 24);
  if ( v29 == 10 || v29 == 32 )
  {
    v37 = *(_QWORD *)(v15 + 88);
    v38 = *(_DWORD *)(v37 + 32);
    v39 = v37 + 24;
    if ( (_DWORD)v79 == 1 )
    {
      if ( v38 > 0x40 )
      {
        v65 = v79;
        v54 = sub_16A57B0(v39);
        v28 = v65;
      }
      else
      {
        v53 = *(_QWORD *)(v37 + 24);
        v54 = v38;
        if ( v53 )
        {
          _BitScanReverse64(&v55, v53);
          v54 = v38 - 64 + (v55 ^ 0x3F);
        }
      }
      v30 = v26 >= v38 - v54;
      goto LABEL_32;
    }
    v40 = *(_QWORD *)(v37 + 24);
    v41 = v38 + 1;
    v42 = 1LL << ((unsigned __int8)v38 - 1);
    if ( v38 > 0x40 )
    {
      if ( (*(_QWORD *)(v40 + 8LL * ((v38 - 1) >> 6)) & v42) == 0 )
      {
        v61 = v79;
        v56 = sub_16A57B0(v39);
        v28 = v61;
        v46 = v38 + 1 - v56;
        goto LABEL_41;
      }
      v60 = v79;
      v43 = sub_16A5810(v39);
      v28 = v60;
      v41 = v38 + 1;
    }
    else
    {
      if ( (v42 & v40) == 0 )
      {
        v46 = 1;
        if ( v40 )
        {
          _BitScanReverse64(&v40, v40);
          v46 = 65 - (v40 ^ 0x3F);
        }
        goto LABEL_41;
      }
      v43 = 64;
      v44 = ~(v40 << (64 - (unsigned __int8)v38));
      if ( v44 )
      {
        _BitScanReverse64(&v45, v44);
        v43 = v45 ^ 0x3F;
      }
    }
    v46 = v41 - v43;
LABEL_41:
    v30 = v26 >= v46;
    goto LABEL_32;
  }
  v63 = v79;
  if ( !(unsigned __int8)sub_21CB9A0(v15, v26, &v81, v27) )
  {
LABEL_9:
    v5 = 0;
    goto LABEL_10;
  }
  v28 = v63;
  v30 = (_DWORD)v79 == (_DWORD)v81;
LABEL_32:
  if ( !v30 )
    goto LABEL_9;
  *((_QWORD *)&v57 + 1) = v67;
  *(_QWORD *)&v57 = v14;
  v64 = v28;
  v66 = (unsigned __int8)(((_BYTE)v73 != 5) + 4);
  v31 = sub_1D309E0(
          *(__int64 **)(a2 + 16),
          145,
          (__int64)&v75,
          v66,
          0,
          0,
          *(double *)v12.m128i_i64,
          *(double *)v13.m128i_i64,
          *(double *)a5.m128i_i64,
          v57);
  v32 = *(__int64 **)(a2 + 16);
  v68.m128i_i64[0] = v15;
  v34 = v33;
  v35 = v31;
  *(_QWORD *)&v36 = sub_1D309E0(
                      v32,
                      145,
                      (__int64)&v75,
                      (unsigned int)v66,
                      0,
                      0,
                      *(double *)v12.m128i_i64,
                      *(double *)v13.m128i_i64,
                      *(double *)a5.m128i_i64,
                      __PAIR128__(v68.m128i_u64[1], v15));
  v5 = sub_1D332F0(
         *(__int64 **)(a2 + 16),
         295 - ((unsigned int)(v64 == 0) - 1),
         (__int64)&v75,
         v73,
         v74,
         0,
         *(double *)v12.m128i_i64,
         *(double *)v13.m128i_i64,
         a5,
         v35,
         v34,
         v36);
LABEL_10:
  if ( v75 )
    sub_161E7C0((__int64)&v75, v75);
  return v5;
}
