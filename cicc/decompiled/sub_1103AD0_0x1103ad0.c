// Function: sub_1103AD0
// Address: 0x1103ad0
//
_QWORD *__fastcall sub_1103AD0(const __m128i *a1, __int64 a2)
{
  int v4; // eax
  __int64 v5; // r14
  int v6; // r13d
  __int64 v7; // r15
  __int64 v9; // rax
  __int64 v10; // rdx
  unsigned __int8 v11; // al
  int v12; // esi
  __int64 v13; // rax
  __int64 *v14; // rsi
  __int64 v15; // rsi
  _BYTE *v16; // r9
  __int64 v17; // rcx
  __int64 v18; // rax
  _QWORD *v19; // r14
  __int64 *v20; // rdi
  __int64 v21; // r10
  __int64 v22; // r8
  _BYTE *v23; // r9
  __int64 v24; // r13
  unsigned int v25; // esi
  unsigned int v26; // edx
  int v27; // r15d
  __m128i v28; // xmm1
  unsigned __int64 v29; // xmm2_8
  __m128i v30; // xmm3
  __int64 v31; // rax
  __int64 v32; // r14
  unsigned int v33; // eax
  __int64 v34; // r10
  __int64 v35; // r11
  __int64 v36; // r14
  __int64 v37; // r8
  unsigned int **v38; // rdi
  __int64 *v39; // rax
  __int64 v40; // rax
  __int64 v41; // r12
  __int64 v42; // r13
  __int64 v43; // rax
  unsigned int *v44; // rax
  __int64 v45; // rax
  _QWORD *v46; // rax
  __int64 v47; // r14
  unsigned int *v48; // r13
  __int64 v49; // rdx
  unsigned int v50; // esi
  __int64 v51; // [rsp+8h] [rbp-138h]
  _BYTE *v52; // [rsp+10h] [rbp-130h]
  __int64 v53; // [rsp+10h] [rbp-130h]
  __int64 v54; // [rsp+10h] [rbp-130h]
  char v55; // [rsp+18h] [rbp-128h]
  _BYTE *v56; // [rsp+20h] [rbp-120h]
  _BYTE *v57; // [rsp+20h] [rbp-120h]
  _BYTE *v58; // [rsp+20h] [rbp-120h]
  unsigned int v59; // [rsp+20h] [rbp-120h]
  __int64 v60; // [rsp+20h] [rbp-120h]
  __int64 v61; // [rsp+20h] [rbp-120h]
  __int64 v62; // [rsp+20h] [rbp-120h]
  unsigned __int8 *v63; // [rsp+28h] [rbp-118h]
  _BYTE *v64; // [rsp+28h] [rbp-118h]
  __int64 v65; // [rsp+28h] [rbp-118h]
  __int64 v66; // [rsp+28h] [rbp-118h]
  unsigned int *v67; // [rsp+28h] [rbp-118h]
  __int64 v68; // [rsp+28h] [rbp-118h]
  int v69; // [rsp+30h] [rbp-110h] BYREF
  unsigned int v70; // [rsp+34h] [rbp-10Ch] BYREF
  __int64 v71; // [rsp+38h] [rbp-108h] BYREF
  __int64 v72; // [rsp+40h] [rbp-100h] BYREF
  __int64 v73; // [rsp+48h] [rbp-F8h] BYREF
  __int64 v74; // [rsp+50h] [rbp-F0h] BYREF
  unsigned int v75; // [rsp+58h] [rbp-E8h]
  _QWORD v76[6]; // [rsp+60h] [rbp-E0h] BYREF
  __int64 v77[4]; // [rsp+90h] [rbp-B0h] BYREF
  __int16 v78; // [rsp+B0h] [rbp-90h]
  __m128i v79[2]; // [rsp+C0h] [rbp-80h] BYREF
  unsigned __int64 v80; // [rsp+E0h] [rbp-60h]
  __int64 v81; // [rsp+E8h] [rbp-58h]
  __m128i v82; // [rsp+F0h] [rbp-50h]
  __int64 v83; // [rsp+100h] [rbp-40h]

  v71 = *(_QWORD *)(a2 + 8);
  v4 = sub_BCB060(v71);
  v5 = *(_QWORD *)(a2 - 32);
  v69 = v4;
  v6 = v4;
  v70 = sub_BCB060(*(_QWORD *)(v5 + 8));
  if ( !v6 )
    return 0;
  if ( (v6 & (v6 - 1)) != 0 )
    return 0;
  v9 = *(_QWORD *)(v5 + 16);
  if ( !v9 )
    return 0;
  v7 = *(_QWORD *)(v9 + 8);
  if ( v7 )
    return 0;
  if ( *(_BYTE *)v5 != 58 )
    return 0;
  v10 = *(_QWORD *)(v5 - 64);
  v11 = *(_BYTE *)v10;
  if ( *(_BYTE *)v10 <= 0x1Cu )
    return 0;
  v12 = v11;
  if ( (unsigned int)v11 - 42 > 0x11 || (unsigned __int8)(**(_BYTE **)(v5 - 32) - 42) > 0x11u )
    return 0;
  v13 = *(_QWORD *)(v10 + 16);
  if ( !v13 || *(_QWORD *)(v13 + 8) || (unsigned int)(v12 - 54) > 1 )
    return (_QWORD *)v7;
  v14 = (*(_BYTE *)(v10 + 7) & 0x40) != 0
      ? *(__int64 **)(v10 - 8)
      : (__int64 *)(v10 - 32LL * (*(_DWORD *)(v10 + 4) & 0x7FFFFFF));
  v15 = *v14;
  v63 = *(unsigned __int8 **)(v5 - 32);
  if ( !v15 )
    return (_QWORD *)v7;
  v72 = v15;
  v56 = (_BYTE *)v10;
  v16 = *(_BYTE **)(sub_986520(v10) + 32);
  if ( !v16 )
    return (_QWORD *)v7;
  v17 = (__int64)v63;
  v18 = *((_QWORD *)v63 + 2);
  if ( !v18 )
    return (_QWORD *)v7;
  v19 = *(_QWORD **)(v18 + 8);
  if ( v19 || (unsigned int)*v63 - 54 > 1 )
    return (_QWORD *)v7;
  v20 = (v63[7] & 0x40) != 0
      ? (__int64 *)*((_QWORD *)v63 - 1)
      : (__int64 *)&v63[-32 * (*((_DWORD *)v63 + 1) & 0x7FFFFFF)];
  v21 = *v20;
  v52 = v16;
  v64 = v56;
  if ( !*v20 )
    return (_QWORD *)v7;
  v73 = *v20;
  v51 = v21;
  v57 = (_BYTE *)v17;
  v22 = *(_QWORD *)(sub_986520(v17) + 32);
  if ( !v22 || *v57 == *v64 )
    return (_QWORD *)v7;
  v23 = v52;
  if ( *v64 == 55 )
  {
    v72 = v51;
    v23 = (_BYTE *)v22;
    v73 = v15;
    v22 = (__int64)v52;
  }
  v76[0] = &v69;
  v76[1] = &v70;
  v76[2] = &v72;
  v58 = (_BYTE *)v22;
  v65 = (__int64)v23;
  v76[3] = &v73;
  v76[4] = a1;
  v55 = 1;
  v24 = sub_1103330((__int64)v76, v23, v22, v6);
  if ( !v24 )
  {
    v24 = sub_1103330((__int64)v76, v58, v65, v69);
    if ( v24 )
    {
      v55 = 0;
      goto LABEL_29;
    }
    return 0;
  }
LABEL_29:
  v25 = v69;
  v26 = v70;
  v75 = v70;
  v27 = v69 - v70;
  if ( v70 > 0x40 )
  {
    sub_C43690((__int64)&v74, 0, 0);
    v26 = v75;
    v25 = v27 + v75;
  }
  else
  {
    v74 = 0;
  }
  if ( v26 != v25 )
  {
    if ( v25 > 0x3F || v26 > 0x40 )
      sub_C43C90(&v74, v25, v26);
    else
      v74 |= 0xFFFFFFFFFFFFFFFFLL >> ((unsigned __int8)v27 + 64) << v25;
  }
  v28 = _mm_loadu_si128(a1 + 7);
  v29 = _mm_loadu_si128(a1 + 8).m128i_u64[0];
  v30 = _mm_loadu_si128(a1 + 9);
  v31 = a1[10].m128i_i64[0];
  v79[0] = _mm_loadu_si128(a1 + 6);
  v80 = v29;
  v83 = v31;
  v81 = a2;
  v79[1] = v28;
  v82 = v30;
  if ( (unsigned __int8)sub_9AC230(v73, (__int64)&v74, v79, 0) )
  {
    v66 = a1[2].m128i_i64[0];
    v78 = 257;
    v32 = *(_QWORD *)(v24 + 8);
    v53 = v71;
    v59 = sub_BCB060(v32);
    v33 = sub_BCB060(v53);
    v34 = v53;
    if ( v59 < v33 )
    {
      if ( v53 != v32 )
      {
        v43 = (*(__int64 (__fastcall **)(_QWORD, __int64, __int64, __int64))(**(_QWORD **)(v66 + 80) + 120LL))(
                *(_QWORD *)(v66 + 80),
                39,
                v24,
                v53);
        if ( v43 )
        {
          v34 = v71;
          v24 = v43;
          v66 = a1[2].m128i_i64[0];
        }
        else
        {
          LOWORD(v80) = 257;
          v46 = sub_BD2C40(72, unk_3F10A14);
          v47 = (__int64)v46;
          if ( v46 )
            sub_B515B0((__int64)v46, v24, v53, (__int64)v79, 0, 0);
          (*(void (__fastcall **)(_QWORD, __int64, __int64 *, _QWORD, _QWORD))(**(_QWORD **)(v66 + 88) + 16LL))(
            *(_QWORD *)(v66 + 88),
            v47,
            v77,
            *(_QWORD *)(v66 + 56),
            *(_QWORD *)(v66 + 64));
          v48 = *(unsigned int **)v66;
          v68 = *(_QWORD *)v66 + 16LL * *(unsigned int *)(v66 + 8);
          while ( (unsigned int *)v68 != v48 )
          {
            v49 = *((_QWORD *)v48 + 1);
            v50 = *v48;
            v48 += 4;
            sub_B99FD0(v47, v50, v49);
          }
          v34 = v71;
          v24 = v47;
          v66 = a1[2].m128i_i64[0];
        }
      }
    }
    else if ( v59 > v33 )
    {
      v45 = sub_A82DA0((unsigned int **)v66, v24, v53, (__int64)v77, 0, 0);
      v34 = v71;
      v24 = v45;
      v66 = a1[2].m128i_i64[0];
    }
    v78 = 257;
    v35 = v72;
    if ( *(_QWORD *)(v72 + 8) == v34 )
    {
      v36 = v72;
    }
    else
    {
      v54 = v34;
      v60 = v72;
      v36 = (*(__int64 (__fastcall **)(_QWORD, __int64, __int64, __int64))(**(_QWORD **)(v66 + 80) + 120LL))(
              *(_QWORD *)(v66 + 80),
              38,
              v72,
              v34);
      if ( !v36 )
      {
        LOWORD(v80) = 257;
        v36 = sub_B51D30(38, v60, v54, (__int64)v79, 0, 0);
        (*(void (__fastcall **)(_QWORD, __int64, __int64 *, _QWORD, _QWORD))(**(_QWORD **)(v66 + 88) + 16LL))(
          *(_QWORD *)(v66 + 88),
          v36,
          v77,
          *(_QWORD *)(v66 + 56),
          *(_QWORD *)(v66 + 64));
        v44 = *(unsigned int **)v66;
        v62 = *(_QWORD *)v66 + 16LL * *(unsigned int *)(v66 + 8);
        if ( *(_QWORD *)v66 != v62 )
        {
          do
          {
            v67 = v44;
            sub_B99FD0(v36, *v44, *((_QWORD *)v44 + 1));
            v44 = v67 + 4;
          }
          while ( (unsigned int *)v62 != v67 + 4 );
        }
      }
      v35 = v72;
    }
    v37 = v36;
    if ( v73 != v35 )
    {
      v38 = (unsigned int **)a1[2].m128i_i64[0];
      LOWORD(v80) = 257;
      v37 = sub_A82DA0(v38, v73, v71, (__int64)v79, 0, 0);
    }
    v61 = v37;
    v39 = (__int64 *)sub_B43CA0(a2);
    v40 = sub_B6E160(v39, (unsigned int)(v55 == 0) + 180, (__int64)&v71, 1);
    v77[2] = v24;
    LOWORD(v80) = 257;
    v41 = v40;
    v42 = 0;
    v77[0] = v36;
    v77[1] = v61;
    if ( v40 )
      v42 = *(_QWORD *)(v40 + 24);
    v19 = sub_BD2CC0(88, 4u);
    if ( v19 )
    {
      sub_B44260((__int64)v19, **(_QWORD **)(v42 + 16), 56, 4u, 0, 0);
      v19[9] = 0;
      sub_B4A290((__int64)v19, v42, v41, v77, 3, (__int64)v79, 0, 0);
    }
  }
  if ( v75 > 0x40 && v74 )
    j_j___libc_free_0_0(v74);
  return v19;
}
