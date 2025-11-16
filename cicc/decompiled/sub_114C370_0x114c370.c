// Function: sub_114C370
// Address: 0x114c370
//
__int64 __fastcall sub_114C370(const __m128i *a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5)
{
  unsigned int v5; // r15d
  __int64 v7; // rbx
  __int64 v8; // rdi
  unsigned int v9; // r13d
  __int64 v10; // r15
  __int64 v12; // rax
  __int64 v13; // rcx
  __int64 v14; // r9
  __int64 v15; // rcx
  _QWORD *v16; // r15
  _QWORD *v17; // rax
  __int64 v18; // rsi
  unsigned __int64 v19; // rdx
  __int64 v20; // rdi
  __int64 v21; // rax
  __int64 v22; // r9
  unsigned __int8 v23; // al
  __m128i v24; // rax
  __int64 v25; // rax
  unsigned __int8 *v26; // rdi
  unsigned int i; // eax
  unsigned __int8 **v28; // rdx
  __int64 v29; // rcx
  __int64 v30; // r8
  __int64 v31; // r9
  unsigned __int8 *v32; // r15
  unsigned __int8 **v33; // rax
  unsigned __int8 **v34; // rdi
  char v35; // al
  char v36; // dl
  unsigned __int8 v37; // al
  char v38; // r15
  unsigned int v39; // eax
  __int64 v40; // rdx
  __m128i v41; // xmm1
  __m128i v42; // xmm2
  __m128i v43; // xmm3
  __int64 v44; // rsi
  __int64 v45; // rax
  unsigned int v46; // edx
  unsigned __int64 v47; // rcx
  __int64 v48; // rdi
  int v49; // edx
  __int64 v50; // r14
  char v51; // r15
  __int64 v52; // rax
  __int64 v53; // rdx
  int v54; // r14d
  bool v55; // r14
  __int64 v56; // rax
  unsigned __int8 *v57; // r14
  unsigned __int64 v58; // rdx
  __int64 v59; // rax
  unsigned __int64 v60; // rdx
  unsigned __int8 *v61; // r14
  __int64 v62; // r14
  unsigned __int8 *v63; // r9
  __int64 v64; // rax
  __int64 v65; // r14
  unsigned __int64 v66; // rdx
  unsigned __int8 **v67; // rdx
  unsigned __int8 v68; // al
  char v69; // r14
  __int64 v70; // rax
  __int64 v71; // rdx
  __int64 v72; // rdx
  __int64 v73; // rax
  __int64 v74; // r13
  __int64 v75; // rax
  __int64 v76; // rax
  unsigned __int64 v77; // rdi
  __int64 v78; // [rsp+8h] [rbp-158h]
  unsigned __int8 *v79; // [rsp+8h] [rbp-158h]
  unsigned __int64 v80; // [rsp+10h] [rbp-150h]
  unsigned int v81; // [rsp+1Ch] [rbp-144h]
  __int64 v82; // [rsp+20h] [rbp-140h]
  char v83; // [rsp+28h] [rbp-138h]
  __int64 v84; // [rsp+28h] [rbp-138h]
  __int64 v85; // [rsp+30h] [rbp-130h]
  int v86; // [rsp+30h] [rbp-130h]
  __int64 v87; // [rsp+30h] [rbp-130h]
  unsigned int v88; // [rsp+30h] [rbp-130h]
  __int64 v91; // [rsp+50h] [rbp-110h] BYREF
  unsigned int v92; // [rsp+58h] [rbp-108h]
  __int64 v93; // [rsp+60h] [rbp-100h] BYREF
  unsigned int v94; // [rsp+68h] [rbp-F8h]
  unsigned __int64 v95; // [rsp+70h] [rbp-F0h] BYREF
  __int64 v96; // [rsp+78h] [rbp-E8h]
  _BYTE *v97; // [rsp+80h] [rbp-E0h] BYREF
  __int64 v98; // [rsp+88h] [rbp-D8h]
  _BYTE v99[32]; // [rsp+90h] [rbp-D0h] BYREF
  unsigned __int8 **v100; // [rsp+B0h] [rbp-B0h] BYREF
  unsigned int v101; // [rsp+B8h] [rbp-A8h]
  unsigned int v102; // [rsp+BCh] [rbp-A4h]
  unsigned __int8 *v103; // [rsp+C0h] [rbp-A0h] BYREF
  unsigned int v104; // [rsp+C8h] [rbp-98h]
  __m128i v105; // [rsp+E0h] [rbp-80h] BYREF
  __m128i v106; // [rsp+F0h] [rbp-70h]
  _OWORD v107[2]; // [rsp+100h] [rbp-60h] BYREF
  __int64 v108; // [rsp+120h] [rbp-40h]

  v5 = *(_DWORD *)(a2 + 4) & 0x7FFFFFF;
  if ( v5 <= 1 )
    return 0;
  v7 = 1;
  while ( 1 )
  {
    v8 = *(_QWORD *)(a2 - 32LL * v5 + 32 * v7);
    if ( *(_BYTE *)v8 != 17 )
      break;
    v9 = *(_DWORD *)(v8 + 32);
    if ( v9 <= 0x40 )
    {
      if ( *(_QWORD *)(v8 + 24) )
        return 0;
    }
    else if ( v9 != (unsigned int)sub_C444A0(v8 + 24) )
    {
      return 0;
    }
    if ( v5 == (_DWORD)++v7 )
      return 0;
  }
  if ( *(_BYTE *)v8 <= 0x15u )
    return 0;
  v98 = 0x400000000LL;
  v12 = 32 * (1LL - v5);
  v13 = 32 * (v7 + 1LL - v5);
  v97 = v99;
  v14 = a2 + v13;
  v15 = v13 - v12;
  v16 = (_QWORD *)(a2 + v12);
  v17 = v99;
  v18 = 0;
  v19 = v15 >> 5;
  if ( (unsigned __int64)v15 > 0x80 )
  {
    v84 = v14;
    v86 = v15 >> 5;
    sub_C8D5F0((__int64)&v97, v99, v19, 8u, a5, v14);
    v18 = (unsigned int)v98;
    v14 = v84;
    LODWORD(v19) = v86;
    v17 = &v97[8 * (unsigned int)v98];
  }
  if ( v16 != (_QWORD *)v14 )
  {
    do
    {
      if ( v17 )
        *v17 = *v16;
      v16 += 4;
      ++v17;
    }
    while ( (_QWORD *)v14 != v16 );
    v18 = (unsigned int)v98;
  }
  v20 = *(_QWORD *)(a2 + 72);
  LODWORD(v98) = v18 + v19;
  if ( sub_BCEA30(v20) )
    goto LABEL_17;
  v18 = (__int64)v97;
  v21 = sub_B4DC50(v20, (__int64)v97, (unsigned int)v98);
  v22 = v21;
  if ( !v21 )
    goto LABEL_17;
  v23 = *(_BYTE *)(v21 + 8);
  if ( v23 != 12 && v23 > 3u && v23 != 5 && (v23 & 0xFD) != 4 && (v23 & 0xFB) != 0xA )
  {
    if ( (unsigned __int8)(v23 - 15) > 3u && v23 != 20 )
      goto LABEL_17;
    v18 = 0;
    v87 = v22;
    v35 = sub_BCEBA0(v22, 0);
    v22 = v87;
    if ( !v35 )
      goto LABEL_17;
  }
  v82 = v22;
  v85 = a1[5].m128i_i64[1];
  v83 = sub_AE5020(v85, v22);
  v18 = v82;
  v24.m128i_i64[0] = sub_9208B0(v85, v82);
  v105 = v24;
  v80 = ((1LL << v83) + ((unsigned __int64)(v24.m128i_i64[0] + 7) >> 3) - 1) >> v83 << v83;
  v81 = v7 + 1;
  v25 = *(_DWORD *)(a2 + 4) & 0x7FFFFFF;
  if ( (_DWORD)v7 + 1 != (_DWORD)v25 )
  {
    if ( !sub_B4DE30(a2) )
      goto LABEL_17;
    v25 = *(_DWORD *)(a2 + 4) & 0x7FFFFFF;
  }
  v105.m128i_i64[0] = 0;
  v106.m128i_i64[0] = 4;
  v26 = *(unsigned __int8 **)(a2 - 32 * v25);
  v106.m128i_i8[12] = 1;
  v106.m128i_i32[2] = 0;
  v102 = 4;
  v103 = v26;
  v105.m128i_i64[1] = (__int64)v107;
  v100 = &v103;
  for ( i = 1; ; v26 = v34[i - 1] )
  {
    v101 = i - 1;
    v32 = sub_BD3990(v26, v18);
    if ( !v106.m128i_i8[12] )
      goto LABEL_44;
    v33 = (unsigned __int8 **)v105.m128i_i64[1];
    v29 = v106.m128i_u32[1];
    v28 = (unsigned __int8 **)(v105.m128i_i64[1] + 8LL * v106.m128i_u32[1]);
    if ( (unsigned __int8 **)v105.m128i_i64[1] != v28 )
    {
      while ( v32 != *v33 )
      {
        if ( v28 == ++v33 )
          goto LABEL_65;
      }
      goto LABEL_31;
    }
LABEL_65:
    if ( v106.m128i_i32[1] < (unsigned __int32)v106.m128i_i32[0] )
    {
      ++v106.m128i_i32[1];
      *v28 = v32;
      ++v105.m128i_i64[0];
    }
    else
    {
LABEL_44:
      v18 = (__int64)v32;
      sub_C8CC70((__int64)&v105, (__int64)v32, (__int64)v28, v29, v30, v31);
      if ( !v36 )
        goto LABEL_31;
    }
    v37 = *v32;
    if ( *v32 <= 0x1Cu )
    {
      if ( v37 != 1 )
      {
        if ( v37 != 3 )
          goto LABEL_48;
        if ( sub_B2FC80((__int64)v32) )
          goto LABEL_48;
        if ( (unsigned __int8)sub_B2F6B0((__int64)v32) )
          goto LABEL_48;
        v68 = v32[80];
        if ( (v68 & 2) != 0 )
          goto LABEL_48;
        if ( (v68 & 1) == 0 )
          goto LABEL_48;
        v18 = *((_QWORD *)v32 + 3);
        v69 = sub_AE5020(v85, v18);
        v70 = sub_9208B0(v85, v18);
        v96 = v71;
        v95 = (((unsigned __int64)(v70 + 7) >> 3) + (1LL << v69) - 1) >> v69 << v69;
        if ( v80 < sub_CA1930(&v95) )
          goto LABEL_48;
        goto LABEL_31;
      }
      if ( (unsigned __int8)sub_B2F6B0((__int64)v32) )
        goto LABEL_48;
      v59 = v101;
      v61 = (unsigned __int8 *)*((_QWORD *)v32 - 4);
      v60 = v101 + 1LL;
      if ( v60 <= v102 )
        goto LABEL_90;
LABEL_109:
      v18 = (__int64)&v103;
      sub_C8D5F0((__int64)&v100, &v103, v60, 8u, v30, v31);
      v59 = v101;
LABEL_90:
      v100[v59] = v61;
      v34 = v100;
      i = ++v101;
      goto LABEL_32;
    }
    if ( v37 == 86 )
    {
      v56 = v101;
      v57 = (unsigned __int8 *)*((_QWORD *)v32 - 8);
      v58 = v101 + 1LL;
      if ( v58 > v102 )
      {
        v18 = (__int64)&v103;
        sub_C8D5F0((__int64)&v100, &v103, v58, 8u, v30, v31);
        v56 = v101;
      }
      v100[v56] = v57;
      v59 = v101 + 1;
      v60 = v59 + 1;
      ++v101;
      v61 = (unsigned __int8 *)*((_QWORD *)v32 - 4);
      if ( v59 + 1 <= (unsigned __int64)v102 )
        goto LABEL_90;
      goto LABEL_109;
    }
    if ( v37 != 84 )
    {
      if ( v37 != 60 )
        goto LABEL_48;
      v48 = *((_QWORD *)v32 + 9);
      v49 = *(unsigned __int8 *)(v48 + 8);
      if ( (_BYTE)v49 != 12 && (unsigned __int8)v49 > 3u && (_BYTE)v49 != 5 && (v49 & 0xFB) != 0xA && (v49 & 0xFD) != 4 )
      {
        if ( (unsigned __int8)(*(_BYTE *)(v48 + 8) - 15) > 3u && v49 != 20 )
          goto LABEL_48;
        v18 = 0;
        if ( !(unsigned __int8)sub_BCEBA0(v48, 0) )
          goto LABEL_48;
      }
      v50 = *((_QWORD *)v32 - 4);
      if ( *(_BYTE *)v50 != 17 )
        goto LABEL_48;
      v78 = *((_QWORD *)v32 + 9);
      v51 = sub_AE5020(v85, v78);
      v52 = sub_9208B0(v85, v78);
      v96 = v53;
      v95 = v52;
      v18 = (((unsigned __int64)(v52 + 7) >> 3) + (1LL << v51) - 1) >> v51 << v51;
      if ( (_BYTE)v53 )
        goto LABEL_48;
      v94 = 128;
      sub_C43690((__int64)&v93, v18, 0);
      sub_C449B0((__int64)&v91, (const void **)(v50 + 24), 0x80u);
      v18 = (__int64)&v91;
      sub_C472A0((__int64)&v95, (__int64)&v91, &v93);
      v54 = v96;
      if ( (unsigned int)v96 <= 0x40 )
      {
        v55 = v80 < v95;
        goto LABEL_79;
      }
      if ( v54 - (unsigned int)sub_C444A0((__int64)&v95) <= 0x40 )
      {
        v77 = v95;
        v55 = v80 < *(_QWORD *)v95;
LABEL_140:
        j_j___libc_free_0_0(v77);
      }
      else
      {
        v77 = v95;
        v55 = 1;
        if ( v95 )
          goto LABEL_140;
      }
LABEL_79:
      if ( v92 > 0x40 && v91 )
        j_j___libc_free_0_0(v91);
      if ( v94 > 0x40 && v93 )
        j_j___libc_free_0_0(v93);
      if ( v55 )
      {
LABEL_48:
        v38 = 0;
        v34 = v100;
        goto LABEL_49;
      }
LABEL_31:
      i = v101;
      v34 = v100;
      goto LABEL_32;
    }
    v62 = 32LL * (*((_DWORD *)v32 + 1) & 0x7FFFFFF);
    if ( (v32[7] & 0x40) != 0 )
    {
      v63 = (unsigned __int8 *)*((_QWORD *)v32 - 1);
      v32 = &v63[v62];
    }
    else
    {
      v63 = &v32[-v62];
    }
    v64 = v101;
    v65 = v62 >> 5;
    v66 = v101 + v65;
    if ( v66 > v102 )
    {
      v18 = (__int64)&v103;
      v79 = v63;
      sub_C8D5F0((__int64)&v100, &v103, v66, 8u, v30, (__int64)v63);
      v64 = v101;
      v63 = v79;
    }
    v34 = v100;
    v67 = &v100[v64];
    if ( v63 != v32 )
    {
      do
      {
        if ( v67 )
          *v67 = *(unsigned __int8 **)v63;
        v63 += 32;
        ++v67;
      }
      while ( v63 != v32 );
      LODWORD(v64) = v101;
      v34 = v100;
    }
    v101 = v64 + v65;
    i = v64 + v65;
LABEL_32:
    if ( !i )
      break;
  }
  v38 = 1;
LABEL_49:
  if ( v34 != &v103 )
    _libc_free(v34, v18);
  if ( !v106.m128i_i8[12] )
    _libc_free(v105.m128i_i64[1], v18);
  if ( !v38 )
  {
LABEL_17:
    if ( v97 != v99 )
      _libc_free(v97, v18);
    return 0;
  }
  v39 = *(_DWORD *)(a2 + 4) & 0x7FFFFFF;
  if ( v81 != v39 )
  {
    v40 = v39;
    v88 = v39 - 1;
    while ( 1 )
    {
      v41 = _mm_loadu_si128(a1 + 7);
      v42 = _mm_loadu_si128(a1 + 8);
      v43 = _mm_loadu_si128(a1 + 9);
      v105 = _mm_loadu_si128(a1 + 6);
      v107[0] = v42;
      v106 = v41;
      v44 = *(_QWORD *)(a2 + 32 * (v81 - v40));
      v45 = a1[10].m128i_i64[0];
      v107[1] = v43;
      v108 = v45;
      *((_QWORD *)&v107[0] + 1) = a3;
      sub_9AC330((__int64)&v100, v44, 0, &v105);
      v46 = v101;
      v18 = 1LL << ((unsigned __int8)v101 - 1);
      v47 = v101 > 0x40 ? (unsigned __int64)v100[(v101 - 1) >> 6] : (unsigned __int64)v100;
      if ( (v47 & v18) == 0 )
        break;
      if ( v104 > 0x40 && v103 )
      {
        j_j___libc_free_0_0(v103);
        v46 = v101;
      }
      if ( v46 > 0x40 && v100 )
        j_j___libc_free_0_0(v100);
      if ( v81 == v88 )
        goto LABEL_125;
      ++v81;
      v40 = *(_DWORD *)(a2 + 4) & 0x7FFFFFF;
    }
    if ( v104 > 0x40 && v103 )
    {
      j_j___libc_free_0_0(v103);
      v46 = v101;
    }
    if ( v46 > 0x40 && v100 )
      j_j___libc_free_0_0(v100);
    goto LABEL_17;
  }
LABEL_125:
  if ( v97 != v99 )
    _libc_free(v97, v18);
  v10 = sub_B47F80((_BYTE *)a2);
  v72 = sub_AD64C0(*(_QWORD *)(*(_QWORD *)(a2 + 32 * (v7 - (*(_DWORD *)(a2 + 4) & 0x7FFFFFF))) + 8LL), 0, 0);
  if ( (*(_BYTE *)(v10 + 7) & 0x40) != 0 )
    v73 = *(_QWORD *)(v10 - 8);
  else
    v73 = v10 - 32LL * (*(_DWORD *)(v10 + 4) & 0x7FFFFFF);
  v74 = v73 + 32 * v7;
  if ( *(_QWORD *)v74 )
  {
    v75 = *(_QWORD *)(v74 + 8);
    **(_QWORD **)(v74 + 16) = v75;
    if ( v75 )
      *(_QWORD *)(v75 + 16) = *(_QWORD *)(v74 + 16);
  }
  *(_QWORD *)v74 = v72;
  if ( v72 )
  {
    v76 = *(_QWORD *)(v72 + 16);
    *(_QWORD *)(v74 + 8) = v76;
    if ( v76 )
      *(_QWORD *)(v76 + 16) = v74 + 8;
    *(_QWORD *)(v74 + 16) = v72 + 16;
    *(_QWORD *)(v72 + 16) = v74;
  }
  sub_B44220((_QWORD *)v10, a2 + 24, 0);
  v105.m128i_i64[0] = v10;
  sub_114BD80(a1[2].m128i_i64[1] + 2096, v105.m128i_i64);
  return v10;
}
