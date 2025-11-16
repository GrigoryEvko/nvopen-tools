// Function: sub_1F75380
// Address: 0x1f75380
//
__int64 *__fastcall sub_1F75380(__int64 a1, __int64 *a2, __m128 a3, __int64 a4, __int64 a5, __int64 a6, __int64 a7)
{
  char *v7; // r12
  char v8; // al
  const void **v9; // rdx
  unsigned int v10; // r15d
  __int64 *v11; // rax
  __m128i v12; // xmm1
  __m128i v13; // xmm2
  __int64 *v14; // rbx
  __int64 v15; // rdx
  char v16; // al
  const void **v17; // rdx
  unsigned int v18; // r13d
  unsigned int v19; // r14d
  int v20; // r12d
  unsigned int v21; // r15d
  unsigned int v22; // r13d
  __int64 v23; // rsi
  unsigned int v24; // eax
  char v25; // dl
  char v26; // cl
  unsigned __int8 *v27; // rax
  __int64 v28; // r8
  unsigned int v29; // ecx
  __int128 v30; // rax
  __int64 v31; // rax
  __int64 *v32; // rax
  _BYTE *v33; // rdx
  __int64 v34; // rax
  __int64 v35; // r15
  _BYTE *v36; // r14
  __int64 v37; // rsi
  __int64 *v38; // rax
  __int64 v39; // rsi
  __int64 *v40; // r14
  unsigned int v41; // eax
  int v42; // edx
  int v44; // edx
  unsigned int v45; // esi
  const __m128i *v46; // rax
  char v47; // al
  __int64 v48; // rdx
  int v49; // eax
  char *v50; // rcx
  char *v51; // rdi
  char *v52; // rax
  __int64 v53; // rdx
  char *v54; // rdx
  const void *v55; // r14
  __int64 v56; // rsi
  __int64 v57; // r9
  unsigned int v58; // edx
  _QWORD *v59; // rbx
  unsigned int v60; // edx
  unsigned int v61; // r14d
  __int64 v62; // rsi
  __int64 *v63; // rax
  __int128 v64; // [rsp-10h] [rbp-140h]
  __int64 v65; // [rsp+10h] [rbp-120h]
  __int128 v66; // [rsp+20h] [rbp-110h]
  const __m128i *v67; // [rsp+20h] [rbp-110h]
  __int64 v68; // [rsp+20h] [rbp-110h]
  __int128 v69; // [rsp+20h] [rbp-110h]
  __int64 v70; // [rsp+28h] [rbp-108h]
  int v72; // [rsp+3Ch] [rbp-F4h]
  _QWORD *v73; // [rsp+40h] [rbp-F0h]
  unsigned __int64 v74; // [rsp+48h] [rbp-E8h]
  __int64 v75; // [rsp+50h] [rbp-E0h]
  unsigned int v77; // [rsp+80h] [rbp-B0h] BYREF
  const void **v78; // [rsp+88h] [rbp-A8h]
  __int64 v79; // [rsp+90h] [rbp-A0h] BYREF
  const void **v80; // [rsp+98h] [rbp-98h]
  __int64 v81; // [rsp+A0h] [rbp-90h] BYREF
  __int64 v82; // [rsp+A8h] [rbp-88h]
  _BYTE *v83; // [rsp+B0h] [rbp-80h] BYREF
  __int64 v84; // [rsp+B8h] [rbp-78h]
  _BYTE v85[112]; // [rsp+C0h] [rbp-70h] BYREF

  v7 = *(char **)(a1 + 40);
  v8 = *v7;
  v9 = (const void **)*((_QWORD *)v7 + 1);
  LOBYTE(v77) = v8;
  v78 = v9;
  if ( v8 )
    v10 = word_42FA680[(unsigned __int8)(v8 - 14)];
  else
    v10 = sub_1F58D30((__int64)&v77);
  v11 = *(__int64 **)(a1 + 32);
  v12 = _mm_loadu_si128((const __m128i *)v11);
  v13 = _mm_loadu_si128((const __m128i *)(v11 + 5));
  v75 = *v11;
  v14 = *(__int64 **)(*v11 + 32);
  v65 = v11[5];
  v83 = v85;
  v84 = 0x400000000LL;
  v15 = *(_QWORD *)(*v14 + 40) + 16LL * *((unsigned int *)v14 + 2);
  v16 = *(_BYTE *)v15;
  v17 = *(const void ***)(v15 + 8);
  LOBYTE(v79) = v16;
  v80 = v17;
  if ( v16 )
    v18 = word_42FA680[(unsigned __int8)(v16 - 14)];
  else
    v18 = sub_1F58D30((__int64)&v79);
  v72 = v10 / v18;
  if ( 2 * v18 == v10 && *(_WORD *)(v65 + 24) == 48 )
  {
    v47 = *v7;
    v48 = *((_QWORD *)v7 + 1);
    LOBYTE(v81) = v47;
    v82 = v48;
    if ( v47 )
    {
      v50 = *(char **)(a1 + 88);
      v51 = &v50[4 * word_42FA680[(unsigned __int8)(v47 - 14)]];
    }
    else
    {
      v49 = sub_1F58D30((__int64)&v81);
      v50 = *(char **)(a1 + 88);
      v51 = &v50[4 * v49];
    }
    v52 = &v50[4 * v18];
    v53 = (v51 - v52) >> 4;
    a6 = (v51 - v52) >> 2;
    if ( v53 > 0 )
    {
      v54 = &v52[16 * v53];
      while ( 1 )
      {
        if ( *(_DWORD *)v52 != -1 )
          goto LABEL_53;
        if ( *((_DWORD *)v52 + 1) != -1 )
          break;
        if ( *((_DWORD *)v52 + 2) != -1 )
        {
          v52 += 8;
          goto LABEL_53;
        }
        if ( *((_DWORD *)v52 + 3) != -1 )
        {
          v52 += 12;
          goto LABEL_53;
        }
        v52 += 16;
        if ( v54 == v52 )
        {
          a6 = (v51 - v52) >> 2;
          goto LABEL_67;
        }
      }
      v52 += 4;
LABEL_53:
      if ( v52 == v51 )
        goto LABEL_54;
      goto LABEL_6;
    }
LABEL_67:
    if ( a6 != 2 )
    {
      if ( a6 != 3 )
      {
        if ( a6 != 1 )
          goto LABEL_54;
        goto LABEL_70;
      }
      if ( *(_DWORD *)v52 != -1 )
        goto LABEL_53;
      v52 += 4;
    }
    if ( *(_DWORD *)v52 != -1 )
      goto LABEL_53;
    v52 += 4;
LABEL_70:
    if ( *(_DWORD *)v52 != -1 )
      goto LABEL_53;
LABEL_54:
    v55 = v50;
    v56 = *(_QWORD *)(a1 + 72);
    v81 = v56;
    if ( v56 )
      sub_1623A60((__int64)&v81, v56, 2);
    LODWORD(v82) = *(_DWORD *)(a1 + 64);
    v73 = sub_1D41320(
            (__int64)a2,
            (unsigned int)v79,
            v80,
            (__int64)&v81,
            *v14,
            v14[1],
            *(double *)a3.m128_u64,
            *(double *)v12.m128i_i64,
            v13,
            v14[5],
            v14[6],
            v55,
            v18);
    v74 = v58 | v12.m128i_i64[1] & 0xFFFFFFFF00000000LL;
    if ( v81 )
      sub_161E7C0((__int64)&v81, v81);
    v81 = 0;
    LODWORD(v82) = 0;
    v59 = sub_1D2B300(a2, 0x30u, (__int64)&v81, v79, (__int64)v80, v57);
    v61 = v60;
    if ( v81 )
      sub_161E7C0((__int64)&v81, v81);
    *(_QWORD *)&v69 = v59;
    v62 = *(_QWORD *)(a1 + 72);
    v81 = v62;
    if ( v62 )
      sub_1623A60((__int64)&v81, v62, 2);
    LODWORD(v82) = *(_DWORD *)(a1 + 64);
    *((_QWORD *)&v69 + 1) = v61 | v13.m128i_i64[1] & 0xFFFFFFFF00000000LL;
    v63 = sub_1D332F0(
            a2,
            107,
            (__int64)&v81,
            v77,
            v78,
            0,
            *(double *)a3.m128_u64,
            *(double *)v12.m128i_i64,
            v13,
            (__int64)v73,
            v74,
            v69);
    v39 = v81;
    v40 = v63;
    if ( !v81 )
      goto LABEL_33;
    goto LABEL_26;
  }
LABEL_6:
  if ( v18 > v10 )
  {
    v33 = v85;
    v34 = 0;
LABEL_23:
    v35 = v34;
    v36 = v33;
    v37 = *(_QWORD *)(a1 + 72);
    v81 = v37;
    if ( v37 )
      sub_1623A60((__int64)&v81, v37, 2);
    *((_QWORD *)&v64 + 1) = v35;
    *(_QWORD *)&v64 = v36;
    LODWORD(v82) = *(_DWORD *)(a1 + 64);
    v38 = sub_1D359D0(a2, 107, (__int64)&v81, v77, v78, 0, *(double *)a3.m128_u64, *(double *)v12.m128i_i64, v13, v64);
    v39 = v81;
    v40 = v38;
    if ( !v81 )
      goto LABEL_33;
LABEL_26:
    sub_161E7C0((__int64)&v81, v39);
    goto LABEL_33;
  }
  v19 = v18;
  v20 = 0;
  v21 = v18;
  v22 = 0;
  while ( 1 )
  {
    v23 = *(_QWORD *)(a1 + 88);
    if ( v19 == v22 )
      break;
    v24 = v22;
    v25 = 1;
    v26 = 1;
    do
    {
      if ( *(int *)(v23 + 4LL * v24) >= 0 )
        v26 = 0;
      else
        v25 = 0;
      ++v24;
    }
    while ( v24 != v19 );
    if ( v25 )
      break;
    if ( !v26 )
      goto LABEL_32;
    v27 = (unsigned __int8 *)(*(_QWORD *)(**(_QWORD **)(v75 + 32) + 40LL)
                            + 16LL * *(unsigned int *)(*(_QWORD *)(v75 + 32) + 8LL));
    v28 = *((_QWORD *)v27 + 1);
    v29 = *v27;
    v81 = 0;
    LODWORD(v82) = 0;
    *(_QWORD *)&v30 = sub_1D2B300(a2, 0x30u, (__int64)&v81, v29, v28, a7);
    a7 = *((_QWORD *)&v30 + 1);
    a6 = v30;
    if ( v81 )
    {
      v66 = v30;
      sub_161E7C0((__int64)&v81, v81);
      a7 = *((_QWORD *)&v66 + 1);
      a6 = v66;
    }
    v31 = (unsigned int)v84;
    if ( (unsigned int)v84 >= HIDWORD(v84) )
    {
      v68 = a6;
      v70 = a7;
      sub_16CD150((__int64)&v83, v85, 0, 16, a6, a7);
      v31 = (unsigned int)v84;
      a7 = v70;
      a6 = v68;
    }
    v32 = (__int64 *)&v83[16 * v31];
    *v32 = a6;
    v32[1] = a7;
    LODWORD(v84) = v84 + 1;
LABEL_21:
    ++v20;
    v22 += v21;
    v19 += v21;
    if ( v72 == v20 )
    {
      v33 = v83;
      v34 = (unsigned int)v84;
      goto LABEL_23;
    }
  }
  v41 = *(_DWORD *)(v23 + 4LL * v22) / v21;
  if ( *(_DWORD *)(v23 + 4LL * v22) % v21 )
    goto LABEL_32;
  if ( v21 == 1 )
  {
LABEL_36:
    v44 = v84;
    v45 = *(_DWORD *)(v75 + 56);
    if ( v41 < v45 )
    {
      v46 = (const __m128i *)(*(_QWORD *)(v75 + 32) + 40LL * v41);
      if ( HIDWORD(v84) > (unsigned int)v84 )
        goto LABEL_38;
    }
    else
    {
      v46 = (const __m128i *)(*(_QWORD *)(v65 + 32) + 40LL * (v41 - v45));
      if ( HIDWORD(v84) > (unsigned int)v84 )
      {
LABEL_38:
        a3 = (__m128)_mm_loadu_si128(v46);
        *(__m128 *)&v83[16 * v44] = a3;
        LODWORD(v84) = v84 + 1;
        goto LABEL_21;
      }
    }
    v67 = v46;
    sub_16CD150((__int64)&v83, v85, 0, 16, a6, a7);
    v44 = v84;
    v46 = v67;
    goto LABEL_38;
  }
  v42 = 1;
  while ( 1 )
  {
    a6 = v42 + v22;
    if ( *(_DWORD *)(v23 + 4LL * (v22 - 1 + v42)) + 1 != *(_DWORD *)(v23 + 4 * a6) )
      break;
    if ( ++v42 == v21 )
      goto LABEL_36;
  }
LABEL_32:
  v40 = 0;
LABEL_33:
  if ( v83 != v85 )
    _libc_free((unsigned __int64)v83);
  return v40;
}
