// Function: sub_3276C70
// Address: 0x3276c70
//
__int64 __fastcall sub_3276C70(__int64 *a1, __int64 a2)
{
  const __m128i *v4; // rax
  unsigned __int16 *v5; // rdx
  __int64 v6; // rsi
  __m128i v7; // xmm1
  __int64 v8; // r13
  __int64 v9; // rax
  unsigned __int16 v10; // r14
  __int64 v11; // rdx
  __int64 v12; // rax
  bool v13; // zf
  __int64 v14; // rdx
  __int64 v15; // rax
  __int64 v16; // rdi
  __int64 v17; // rax
  __int64 v18; // r13
  __int64 v20; // rax
  int v21; // eax
  __int64 v22; // rcx
  __int64 v23; // rax
  __int64 v24; // rax
  int v25; // r9d
  __int64 v26; // rax
  __int64 v27; // r12
  __int128 v28; // rax
  __int64 v29; // rax
  __int64 v30; // r14
  __int64 v31; // r10
  __int64 v32; // rdx
  __int64 v33; // r11
  int v34; // edx
  int v35; // ebx
  __int64 v36; // rsi
  __int16 v37; // ax
  int v38; // esi
  __int64 v39; // rax
  __int64 v40; // rax
  __int64 v41; // r12
  __m128i v42; // rax
  __int64 v43; // rdi
  __int64 v44; // rax
  int v45; // ebx
  unsigned __int64 v46; // rdx
  unsigned __int64 v47; // rcx
  __int64 v48; // rdx
  __int64 *v49; // rax
  int v50; // r13d
  __int64 v51; // r10
  __int64 v52; // r11
  __int64 v53; // rax
  __int16 v54; // si
  __int64 v55; // rax
  int v56; // esi
  bool v57; // al
  bool v58; // al
  __int128 v59; // [rsp-20h] [rbp-C0h]
  __int64 v60; // [rsp+8h] [rbp-98h]
  int v61; // [rsp+8h] [rbp-98h]
  __int64 v62; // [rsp+10h] [rbp-90h]
  __int64 v63; // [rsp+10h] [rbp-90h]
  __int64 v64; // [rsp+10h] [rbp-90h]
  __int64 v65; // [rsp+18h] [rbp-88h]
  __int64 v66; // [rsp+18h] [rbp-88h]
  unsigned __int16 v67; // [rsp+20h] [rbp-80h]
  __int128 v68; // [rsp+20h] [rbp-80h]
  __int64 v69; // [rsp+20h] [rbp-80h]
  unsigned __int64 v70; // [rsp+28h] [rbp-78h]
  __m128i v71; // [rsp+30h] [rbp-70h] BYREF
  __int64 v72; // [rsp+40h] [rbp-60h] BYREF
  __int64 v73; // [rsp+48h] [rbp-58h]
  __int64 v74; // [rsp+50h] [rbp-50h] BYREF
  int v75; // [rsp+58h] [rbp-48h]
  __m128i v76; // [rsp+60h] [rbp-40h] BYREF

  v4 = *(const __m128i **)(a2 + 40);
  v5 = *(unsigned __int16 **)(a2 + 48);
  v6 = *(_QWORD *)(a2 + 80);
  v7 = _mm_loadu_si128(v4);
  v8 = v4->m128i_i64[0];
  v9 = v4->m128i_u32[2];
  v10 = *v5;
  v11 = *((_QWORD *)v5 + 1);
  v71 = v7;
  v60 = v9;
  v62 = 16 * v9;
  v12 = *(_QWORD *)(v8 + 48);
  LOWORD(v72) = v10;
  LOWORD(v12) = *(_WORD *)(v12 + v62);
  v73 = v11;
  v74 = v6;
  v67 = v12;
  if ( v6 )
    sub_B96E90((__int64)&v74, v6, 1);
  v13 = *(_DWORD *)(v8 + 24) == 51;
  v75 = *(_DWORD *)(a2 + 72);
  if ( v13 )
  {
    v18 = sub_33FE730(*a1, &v74, (unsigned int)v72, v73, 0, 0.0);
    goto LABEL_9;
  }
  if ( *((_BYTE *)a1 + 33) )
  {
    v14 = a1[1];
    v15 = 1;
    if ( v10 != 1 )
    {
      if ( !v10 )
        goto LABEL_14;
      v15 = v10;
      if ( !*(_QWORD *)(v14 + 8LL * v10 + 112) )
        goto LABEL_14;
    }
    if ( (*(_BYTE *)(v14 + 500 * v15 + 6426) & 0xFB) != 0 )
      goto LABEL_14;
  }
  v16 = *a1;
  v76 = _mm_load_si128(&v71);
  v17 = sub_3402EA0(v16, 220, (unsigned int)&v74, v72, v73, 0, (__int64)&v76, 1);
  if ( v17 )
  {
    v18 = v17;
    goto LABEL_9;
  }
  v14 = a1[1];
  if ( *((_BYTE *)a1 + 33) )
  {
LABEL_14:
    if ( v67 == 1 )
    {
      v20 = 1;
      if ( !*(_BYTE *)(v14 + 7134) )
        goto LABEL_20;
    }
    else
    {
      if ( !v67 )
        goto LABEL_20;
      v20 = v67;
      if ( !*(_QWORD *)(v14 + 8LL * v67 + 112)
        || !*(_BYTE *)(v14 + 500LL * v67 + 6634)
        || !*(_QWORD *)(v14 + 8 * (v67 + 14LL)) )
      {
        goto LABEL_20;
      }
    }
    if ( *(_BYTE *)(v14 + 500 * v20 + 6635) )
      goto LABEL_20;
  }
  else
  {
    if ( v67 == 1 )
    {
      v24 = 1;
      if ( (*(_BYTE *)(v14 + 7134) & 0xFB) == 0 )
        goto LABEL_20;
    }
    else
    {
      if ( !v67 )
        goto LABEL_20;
      v24 = v67;
      if ( !*(_QWORD *)(v14 + 8LL * v67 + 112)
        || (*(_BYTE *)(v14 + 500LL * v67 + 6634) & 0xFB) == 0
        || !*(_QWORD *)(v14 + 8 * (v67 + 14LL)) )
      {
        goto LABEL_20;
      }
    }
    if ( (*(_BYTE *)(v14 + 500 * v24 + 6635) & 0xFB) != 0 )
      goto LABEL_20;
  }
  if ( (unsigned __int8)sub_33DD2A0(*a1, v71.m128i_i64[0], v71.m128i_i64[1], 0) )
  {
    v18 = sub_33FAF80(*a1, 221, (unsigned int)&v74, v72, v73, v25, *(_OWORD *)&v71);
    goto LABEL_9;
  }
LABEL_20:
  v21 = *(_DWORD *)(v8 + 24);
  if ( v21 != 208 )
  {
    if ( v21 != 214 || *(_DWORD *)(**(_QWORD **)(v8 + 40) + 24LL) != 208 )
      goto LABEL_23;
    if ( v10 )
    {
      if ( (unsigned __int16)(v10 - 17) > 0xD3u )
        goto LABEL_55;
    }
    else if ( !sub_30070B0((__int64)&v72) )
    {
LABEL_55:
      if ( !*((_BYTE *)a1 + 33)
        || ((v22 = a1[1], v40 = 1, v10 == 1) || v10 && (v40 = v10, *(_QWORD *)(v22 + 8LL * v10 + 112)))
        && (*(_BYTE *)(v22 + 500 * v40 + 6426) & 0xFB) == 0 )
      {
        v41 = *a1;
        v42.m128i_i64[0] = sub_33FE730(*a1, &v74, (unsigned int)v72, v73, 0, 0.0);
        v43 = *a1;
        v71 = v42;
        v44 = sub_33FE730(v43, &v74, (unsigned int)v72, v73, 0, 1.0);
        v45 = v73;
        v47 = v46;
        v48 = v44;
        v49 = *(__int64 **)(v8 + 40);
        v50 = v72;
        v51 = *v49;
        v52 = v49[1];
        v53 = *(_QWORD *)(*v49 + 48) + 16LL * *((unsigned int *)v49 + 2);
        v54 = *(_WORD *)v53;
        v55 = *(_QWORD *)(v53 + 8);
        v76.m128i_i16[0] = v54;
        v76.m128i_i64[1] = v55;
        if ( v54 )
        {
          v56 = ((unsigned __int16)(v54 - 17) < 0xD4u) + 205;
        }
        else
        {
          v64 = v51;
          v66 = v52;
          v69 = v48;
          v70 = v47;
          v58 = sub_30070B0((__int64)&v76);
          v51 = v64;
          v52 = v66;
          v48 = v69;
          v47 = v70;
          v56 = 205 - (!v58 - 1);
        }
        v39 = sub_340EC60(v41, v56, (unsigned int)&v74, v50, v45, 0, v51, v52, __PAIR128__(v47, v48), *(_OWORD *)&v71);
LABEL_52:
        v18 = v39;
        goto LABEL_9;
      }
      goto LABEL_24;
    }
LABEL_23:
    v22 = a1[1];
    goto LABEL_24;
  }
  if ( *(_WORD *)(*(_QWORD *)(v8 + 48) + v62) != 2 )
    goto LABEL_23;
  if ( v10 )
  {
    if ( (unsigned __int16)(v10 - 17) <= 0xD3u )
      goto LABEL_23;
  }
  else if ( sub_30070B0((__int64)&v72) )
  {
    goto LABEL_23;
  }
  if ( !*((_BYTE *)a1 + 33)
    || ((v22 = a1[1], v26 = 1, v10 == 1) || v10 && (v26 = v10, *(_QWORD *)(v22 + 8LL * v10 + 112)))
    && (*(_BYTE *)(v22 + 500 * v26 + 6426) & 0xFB) == 0 )
  {
    v27 = *a1;
    *(_QWORD *)&v28 = sub_33FE730(*a1, &v74, (unsigned int)v72, v73, 0, 0.0);
    v68 = v28;
    v29 = sub_33FE730(*a1, &v74, (unsigned int)v72, v73, 0, -1.0);
    v30 = *(_QWORD *)(v8 + 48) + v62;
    v31 = v29;
    v33 = v32;
    v71.m128i_i64[0] = v8;
    v34 = v72;
    v35 = v73;
    v36 = *(_QWORD *)(v30 + 8);
    v71.m128i_i64[1] = v60 | v71.m128i_i64[1] & 0xFFFFFFFF00000000LL;
    v37 = *(_WORD *)v30;
    v76.m128i_i64[1] = v36;
    v76.m128i_i16[0] = v37;
    if ( v37 )
    {
      v38 = ((unsigned __int16)(v37 - 17) < 0xD4u) + 205;
    }
    else
    {
      v61 = v72;
      v63 = v31;
      v65 = v33;
      v57 = sub_30070B0((__int64)&v76);
      v34 = v61;
      v31 = v63;
      v33 = v65;
      v38 = 205 - (!v57 - 1);
    }
    *((_QWORD *)&v59 + 1) = v33;
    *(_QWORD *)&v59 = v31;
    v39 = sub_340EC60(v27, v38, (unsigned int)&v74, v34, v35, 0, v71.m128i_i64[0], v71.m128i_i64[1], v59, v68);
    goto LABEL_52;
  }
LABEL_24:
  v18 = 0;
  v23 = sub_3261A10(a2, (int)&v74, *a1, v22);
  if ( v23 )
    v18 = v23;
LABEL_9:
  if ( v74 )
    sub_B91220((__int64)&v74, v74);
  return v18;
}
