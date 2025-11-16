// Function: sub_C29010
// Address: 0xc29010
//
__int64 __fastcall sub_C29010(_QWORD *a1, __int64 a2, unsigned __int8 a3, unsigned int a4)
{
  __int64 v4; // r15
  _QWORD *v5; // r14
  __int64 v7; // rdi
  __int64 v8; // r13
  unsigned __int64 v9; // r12
  unsigned int v10; // ebx
  __int64 *v12; // rax
  __int64 v13; // r13
  unsigned __int64 v14; // rcx
  __m128i **v15; // rax
  __m128i *v16; // rbx
  __int64 v17; // rax
  bool v18; // zf
  unsigned __int8 v19; // bl
  unsigned int v20; // ecx
  __int64 v21; // rsi
  unsigned int v22; // ebx
  int v23; // r13d
  __int64 *v24; // rax
  __int32 v25; // ecx
  unsigned int v26; // r8d
  __int64 v27; // rdi
  __int64 v28; // rax
  __m128i *v29; // rdx
  __int64 v30; // rdi
  __m128i si128; // xmm0
  __int64 v32; // rdi
  _BYTE *v33; // rax
  __int64 v34; // rsi
  __int64 v35; // rax
  unsigned __int64 *v36; // rax
  __int64 v37; // rax
  _QWORD *v38; // rdi
  _QWORD *v39; // rax
  __int64 *v40; // r13
  __int64 v41; // r12
  __int64 v42; // rbx
  unsigned __int64 v43; // r14
  __int64 m128i_i64; // r12
  __int64 v45; // rax
  int v46; // r12d
  unsigned int v47; // eax
  int v48; // [rsp+20h] [rbp-210h]
  unsigned int v49; // [rsp+24h] [rbp-20Ch]
  unsigned int v50; // [rsp+24h] [rbp-20Ch]
  __m128i *v51; // [rsp+28h] [rbp-208h]
  unsigned __int32 v53; // [rsp+38h] [rbp-1F8h]
  __int32 v54; // [rsp+38h] [rbp-1F8h]
  unsigned int v55; // [rsp+3Ch] [rbp-1F4h]
  __m128i *v56; // [rsp+40h] [rbp-1F0h]
  _QWORD *v57; // [rsp+48h] [rbp-1E8h]
  __int64 v58; // [rsp+58h] [rbp-1D8h]
  __int64 v59; // [rsp+60h] [rbp-1D0h]
  __int64 v60; // [rsp+60h] [rbp-1D0h]
  unsigned int v61; // [rsp+60h] [rbp-1D0h]
  bool v63; // [rsp+7Fh] [rbp-1B1h] BYREF
  unsigned int v64; // [rsp+80h] [rbp-1B0h] BYREF
  unsigned int v65; // [rsp+84h] [rbp-1ACh] BYREF
  unsigned int v66; // [rsp+88h] [rbp-1A8h] BYREF
  unsigned int v67; // [rsp+8Ch] [rbp-1A4h] BYREF
  unsigned int v68; // [rsp+90h] [rbp-1A0h] BYREF
  int v69; // [rsp+94h] [rbp-19Ch] BYREF
  unsigned __int64 v70; // [rsp+98h] [rbp-198h] BYREF
  __int64 *v71[2]; // [rsp+A0h] [rbp-190h] BYREF
  __m128i v72[10]; // [rsp+B0h] [rbp-180h] BYREF
  __m128i v73[14]; // [rsp+150h] [rbp-E0h] BYREF

  v4 = (__int64)(a1 + 26);
  v5 = a1;
  v7 = *(unsigned int *)(a2 + 8);
  if ( (_DWORD)v7 )
  {
    v9 = 0;
  }
  else
  {
    if ( (unsigned __int64)(v5[29] + 4LL) > v5[27] )
    {
      v28 = sub_CB72A0(v7, a2);
      v29 = *(__m128i **)(v28 + 32);
      v30 = v28;
      if ( *(_QWORD *)(v28 + 24) - (_QWORD)v29 <= 0x20u )
      {
        v30 = sub_CB6200(v28, "unexpected end of memory buffer: ", 33);
      }
      else
      {
        si128 = _mm_load_si128((const __m128i *)&xmmword_3F64EF0);
        v29[2].m128i_i8[0] = 32;
        *v29 = si128;
        v29[1] = _mm_load_si128((const __m128i *)&xmmword_3F64F00);
        *(_QWORD *)(v28 + 32) += 33LL;
      }
      v32 = sub_CB59D0(v30, v5[29]);
      v33 = *(_BYTE **)(v32 + 32);
      if ( *(_BYTE **)(v32 + 24) == v33 )
      {
        sub_CB6200(v32, "\n", 1);
      }
      else
      {
        *v33 = 10;
        ++*(_QWORD *)(v32 + 32);
      }
      goto LABEL_7;
    }
    v8 = (unsigned int)sub_C5F610(v4, v5 + 29, v5 + 30);
    if ( !(unsigned __int8)sub_C1FF60(v4, v73) )
    {
LABEL_7:
      sub_C1AFD0();
      return 4;
    }
    v9 = v8 | ((unsigned __int64)v73[0].m128i_u32[0] << 32);
  }
  if ( !(unsigned __int8)sub_C1FF60(v4, &v64) )
    goto LABEL_7;
  v12 = (__int64 *)(v5[33] + 32LL * v64);
  v13 = *v12;
  v59 = v12[1];
  if ( !(unsigned __int8)sub_C1FF60(v4, &v65) || !(unsigned __int8)sub_C1FF60(v4, &v66) )
    goto LABEL_7;
  if ( *(_DWORD *)(a2 + 8) )
  {
    v38 = **(_QWORD ***)a2;
    v72[0].m128i_i32[1] = (unsigned __int16)a4;
    v72[0].m128i_i32[0] = HIWORD(a4);
    v39 = (_QWORD *)sub_C273A0(v38, (unsigned int *)v72);
    v73[0].m128i_i64[0] = v13;
    v73[0].m128i_i64[1] = v59;
    v56 = (__m128i *)sub_C27720(v39, v73);
  }
  else
  {
    memset(v73, 0, 0xB0u);
    v14 = v59;
    v73[6].m128i_i64[0] = (__int64)v73[5].m128i_i64;
    v73[6].m128i_i64[1] = (__int64)v73[5].m128i_i64;
    v73[9].m128i_i64[0] = (__int64)v73[8].m128i_i64;
    v73[9].m128i_i64[1] = (__int64)v73[8].m128i_i64;
    if ( v13 )
    {
      sub_C7D030(v72);
      sub_C7D280(v72, v13, v59);
      sub_C7D290(v72, v71);
      v14 = (unsigned __int64)v71[0];
    }
    v70 = v14;
    v15 = (__m128i **)sub_C1DD00(v5 + 1, v14 % v5[2], &v70, v14);
    if ( !v15 || (v16 = *v15) == 0 )
    {
      v71[0] = (__int64 *)&v70;
      v72[0].m128i_i64[0] = (__int64)v73;
      v16 = sub_C286D0(v5 + 1, v71, (const __m128i **)v72);
    }
    v56 = v16 + 1;
    sub_C1FCF0((__int64)v73);
    v17 = sub_C1B1E0(v9, 1u, v16[5].m128i_u64[0], (bool *)v73[0].m128i_i8);
    v18 = v16[4].m128i_i64[1] == 0;
    v16[5].m128i_i64[0] = v17;
    v19 = a3;
    if ( !v18 )
      v19 = 0;
    a3 = v19;
  }
  v20 = v65;
  v56[1].m128i_i64[0] = v13;
  v56[1].m128i_i64[1] = v59;
  v56[2].m128i_i64[0] = 0;
  v56[2].m128i_i64[1] = 0;
  v56[3].m128i_i32[0] = 0;
  if ( !v20 )
  {
LABEL_82:
    if ( !v66 )
    {
LABEL_89:
      sub_C1AFD0();
      return 0;
    }
    v46 = 0;
    while ( (unsigned __int8)sub_C1FF60(v4, v72) )
    {
      v73[0].m128i_i64[0] = (__int64)v73[1].m128i_i64;
      v73[1].m128i_i64[0] = (__int64)v56;
      v73[0].m128i_i64[1] = 0xA00000001LL;
      sub_C1EC40(
        (__int64)v73,
        &v73[1].m128i_i8[8],
        *(char **)a2,
        (char *)(*(_QWORD *)a2 + 8LL * *(unsigned int *)(a2 + 8)));
      v21 = (__int64)v73;
      v47 = sub_C29010(v5, v73, a3, v72[0].m128i_u32[0]);
      if ( v47 )
      {
        v27 = v73[0].m128i_i64[0];
        v10 = v47;
        if ( (__m128i *)v73[0].m128i_i64[0] != &v73[1] )
          goto LABEL_37;
        return v10;
      }
      if ( (__m128i *)v73[0].m128i_i64[0] != &v73[1] )
        _libc_free(v73[0].m128i_i64[0], v73);
      if ( v66 <= ++v46 )
        goto LABEL_89;
    }
    goto LABEL_7;
  }
  v57 = v5;
  v48 = 0;
  v51 = v56 + 5;
  while ( 1 )
  {
    if ( !(unsigned __int8)sub_C1FF60(v4, &v67)
      || !(unsigned __int8)sub_C1FF60(v4, &v68)
      || !(unsigned __int8)sub_C20040(v4, &v70) )
    {
      goto LABEL_7;
    }
    v21 = (__int64)&v73[1].m128i_i64[1];
    v55 = (unsigned __int16)v67;
    v22 = HIWORD(v67);
    v73[0].m128i_i64[0] = (__int64)v73[1].m128i_i64;
    v73[1].m128i_i64[0] = (__int64)v56;
    v73[0].m128i_i64[1] = 0xA00000001LL;
    sub_C1EC40(
      (__int64)v73,
      &v73[1].m128i_i8[8],
      *(char **)a2,
      (char *)(*(_QWORD *)a2 + 8LL * *(unsigned int *)(a2 + 8)));
    if ( a3 )
    {
      v40 = (__int64 *)v73[0].m128i_i64[0];
      v41 = v73[0].m128i_i64[0] + 8LL * v73[0].m128i_u32[2];
      if ( v41 != v73[0].m128i_i64[0] )
      {
        v61 = v22;
        do
        {
          v42 = *v40++;
          *(_QWORD *)(v42 + 56) = sub_C1B1E0(v70, 1u, *(_QWORD *)(v42 + 56), (bool *)v72[0].m128i_i8);
        }
        while ( (__int64 *)v41 != v40 );
        v22 = v61;
      }
      v72[0].m128i_i64[0] = __PAIR64__(v55, v22);
      v43 = v70;
      m128i_i64 = (__int64)v56[5].m128i_i64;
      v45 = v56[5].m128i_i64[1];
      if ( !v45 )
        goto LABEL_92;
      do
      {
        if ( v22 > *(_DWORD *)(v45 + 32) || v22 == *(_DWORD *)(v45 + 32) && v55 > *(_DWORD *)(v45 + 36) )
        {
          v45 = *(_QWORD *)(v45 + 24);
        }
        else
        {
          m128i_i64 = v45;
          v45 = *(_QWORD *)(v45 + 16);
        }
      }
      while ( v45 );
      if ( v51 == (__m128i *)m128i_i64
        || v22 < *(_DWORD *)(m128i_i64 + 32)
        || v22 == *(_DWORD *)(m128i_i64 + 32) && v55 < *(_DWORD *)(m128i_i64 + 36) )
      {
LABEL_92:
        v71[0] = (__int64 *)v72;
        m128i_i64 = sub_C272A0(&v56[4].m128i_i64[1], m128i_i64, v71);
      }
      v21 = 1;
      *(_QWORD *)(m128i_i64 + 40) = sub_C1B1E0(v43, 1u, *(_QWORD *)(m128i_i64 + 40), (bool *)v71);
    }
    if ( v68 )
      break;
LABEL_78:
    if ( (__m128i *)v73[0].m128i_i64[0] != &v73[1] )
      _libc_free(v73[0].m128i_i64[0], v21);
    if ( v65 <= ++v48 )
    {
      v5 = v57;
      goto LABEL_82;
    }
  }
  v23 = 0;
LABEL_32:
  v21 = (__int64)&v69;
  if ( !(unsigned __int8)sub_C1FF60(v4, &v69) )
  {
LABEL_35:
    v10 = 4;
    sub_C1AFD0();
    goto LABEL_36;
  }
  if ( v69 == 7 )
  {
    v21 = (__int64)v71;
    if ( !(unsigned __int8)sub_C1FF60(v4, v71) )
      goto LABEL_35;
    v21 = (__int64)v72;
    if ( !(unsigned __int8)sub_C1FF60(v4, v72) )
      goto LABEL_35;
    v21 = (__int64)v71;
    v24 = (__int64 *)(v57[33] + 32 * (LODWORD(v71[0]) | ((unsigned __int64)v72[0].m128i_u32[0] << 32)));
    v60 = *v24;
    v58 = v24[1];
    if ( !(unsigned __int8)sub_C1FF60(v4, v71) )
      goto LABEL_35;
    v21 = (__int64)v72;
    if ( !(unsigned __int8)sub_C1FF60(v4, v72) )
      goto LABEL_35;
    v25 = v72[0].m128i_i32[0];
    v26 = (unsigned int)v71[0];
    if ( !a3 )
      goto LABEL_31;
    v71[0] = (__int64 *)__PAIR64__(v55, v22);
    v34 = (__int64)v56[5].m128i_i64;
    v35 = v56[5].m128i_i64[1];
    if ( !v35 )
      goto LABEL_56;
    while ( 1 )
    {
      while ( v22 > *(_DWORD *)(v35 + 32) )
      {
        v35 = *(_QWORD *)(v35 + 24);
LABEL_48:
        if ( !v35 )
        {
LABEL_49:
          if ( (__m128i *)v34 == v51
            || v22 < *(_DWORD *)(v34 + 32)
            || v22 == *(_DWORD *)(v34 + 32) && v55 < *(_DWORD *)(v34 + 36) )
          {
LABEL_56:
            v50 = v26;
            v54 = v72[0].m128i_i32[0];
            v72[0].m128i_i64[0] = (__int64)v71;
            v37 = sub_C272A0(&v56[4].m128i_i64[1], v34, (__int64 **)v72);
            v26 = v50;
            v25 = v54;
            v34 = v37;
          }
          v49 = v26;
          v53 = v25;
          v72[0].m128i_i64[0] = v60;
          v72[0].m128i_i64[1] = v58;
          v36 = sub_C1CD30((_QWORD *)(v34 + 48), v72);
          v21 = 1;
          *v36 = sub_C1B1E0(((unsigned __int64)v53 << 32) | v49, 1u, *v36, &v63);
LABEL_31:
          if ( v68 <= ++v23 )
            goto LABEL_78;
          goto LABEL_32;
        }
      }
      if ( v22 == *(_DWORD *)(v35 + 32) && v55 > *(_DWORD *)(v35 + 36) )
      {
        v35 = *(_QWORD *)(v35 + 24);
        goto LABEL_48;
      }
      v34 = v35;
      v35 = *(_QWORD *)(v35 + 16);
      if ( !v35 )
        goto LABEL_49;
    }
  }
  v10 = 5;
  sub_C1AFD0();
LABEL_36:
  v27 = v73[0].m128i_i64[0];
  if ( (__m128i *)v73[0].m128i_i64[0] != &v73[1] )
LABEL_37:
    _libc_free(v27, v21);
  return v10;
}
