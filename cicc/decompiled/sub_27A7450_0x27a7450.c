// Function: sub_27A7450
// Address: 0x27a7450
//
void __fastcall sub_27A7450(__int64 a1, __int64 a2, __int64 a3)
{
  __int64 v3; // rax
  __int64 v4; // rdx
  unsigned __int64 v5; // rax
  unsigned __int64 v6; // rax
  unsigned __int64 v7; // rax
  unsigned __int64 v8; // rax
  unsigned __int64 v9; // rax
  unsigned __int64 v10; // rax
  unsigned __int64 v11; // rax
  __int64 v12; // rcx
  __int64 v13; // r8
  __int64 v14; // r9
  unsigned __int64 v15; // rax
  unsigned __int64 v16; // rax
  unsigned __int64 v17; // rax
  unsigned __int64 v18; // rdi
  unsigned __int64 v19; // rdi
  unsigned __int64 v20; // rdi
  const __m128i *v21; // rsi
  __int64 v22; // rdx
  __int64 v23; // r8
  __int64 v24; // r9
  __int64 v25; // rcx
  const __m128i *v26; // rdi
  unsigned __int64 v27; // r14
  __int64 v28; // rax
  unsigned __int64 v29; // rsi
  __m128i *v30; // rdx
  const __m128i *v31; // rax
  __int64 v32; // r8
  __int64 v33; // r9
  const __m128i *v34; // rcx
  unsigned __int64 v35; // r14
  __int64 v36; // rax
  unsigned __int64 v37; // rdi
  __m128i *v38; // rdx
  const __m128i *v39; // rax
  unsigned __int64 v40; // rax
  unsigned __int64 v41; // r14
  __int64 *v42; // rax
  __int64 v43; // r15
  __int64 v44; // rsi
  __int64 v45; // rdx
  __int64 v46; // r15
  __int64 v47; // r14
  unsigned __int64 v48; // rdi
  __int64 v49; // r13
  __int64 *v50; // rax
  __int64 v51; // rcx
  __int64 *v52; // rdx
  __int64 v53; // r15
  __int64 *v54; // rax
  unsigned __int64 v55; // rdx
  char v56; // cl
  char v57; // dl
  unsigned __int64 v61[16]; // [rsp+30h] [rbp-320h] BYREF
  __m128i v62; // [rsp+B0h] [rbp-2A0h] BYREF
  __int64 v63; // [rsp+C0h] [rbp-290h]
  unsigned int v64; // [rsp+C8h] [rbp-288h]
  char v65; // [rsp+CCh] [rbp-284h]
  _QWORD v66[8]; // [rsp+D0h] [rbp-280h] BYREF
  unsigned __int64 v67; // [rsp+110h] [rbp-240h] BYREF
  unsigned __int64 v68; // [rsp+118h] [rbp-238h]
  unsigned __int64 v69; // [rsp+120h] [rbp-230h]
  __int64 v70; // [rsp+130h] [rbp-220h] BYREF
  __int64 *v71; // [rsp+138h] [rbp-218h]
  unsigned int v72; // [rsp+140h] [rbp-210h]
  unsigned int v73; // [rsp+144h] [rbp-20Ch]
  char v74; // [rsp+14Ch] [rbp-204h]
  _BYTE v75[64]; // [rsp+150h] [rbp-200h] BYREF
  unsigned __int64 v76; // [rsp+190h] [rbp-1C0h] BYREF
  unsigned __int64 v77; // [rsp+198h] [rbp-1B8h]
  unsigned __int64 v78; // [rsp+1A0h] [rbp-1B0h]
  char v79[8]; // [rsp+1B0h] [rbp-1A0h] BYREF
  unsigned __int64 v80; // [rsp+1B8h] [rbp-198h]
  char v81; // [rsp+1CCh] [rbp-184h]
  _BYTE v82[64]; // [rsp+1D0h] [rbp-180h] BYREF
  unsigned __int64 v83; // [rsp+210h] [rbp-140h]
  unsigned __int64 v84; // [rsp+218h] [rbp-138h]
  unsigned __int64 v85; // [rsp+220h] [rbp-130h]
  __m128i v86; // [rsp+230h] [rbp-120h] BYREF
  char v87; // [rsp+240h] [rbp-110h]
  char v88; // [rsp+24Ch] [rbp-104h]
  char v89[64]; // [rsp+250h] [rbp-100h] BYREF
  const __m128i *v90; // [rsp+290h] [rbp-C0h]
  __int64 v91; // [rsp+298h] [rbp-B8h]
  unsigned __int64 v92; // [rsp+2A0h] [rbp-B0h]
  char v93[8]; // [rsp+2A8h] [rbp-A8h] BYREF
  unsigned __int64 v94; // [rsp+2B0h] [rbp-A0h]
  char v95; // [rsp+2C4h] [rbp-8Ch]
  char v96[64]; // [rsp+2C8h] [rbp-88h] BYREF
  const __m128i *v97; // [rsp+308h] [rbp-48h]
  const __m128i *v98; // [rsp+310h] [rbp-40h]
  unsigned __int64 v99; // [rsp+318h] [rbp-38h]

  v3 = *(_QWORD *)(a1 + 224);
  if ( !*(_DWORD *)(v3 + 56) )
    return;
  v4 = **(_QWORD **)(v3 + 48);
  if ( !v4 )
    return;
  memset(v61, 0, 0x78u);
  v62.m128i_i64[1] = (__int64)v66;
  v61[1] = (unsigned __int64)&v61[4];
  v63 = 0x100000008LL;
  v66[0] = v4;
  v86.m128i_i64[0] = v4;
  LODWORD(v61[2]) = 8;
  BYTE4(v61[3]) = 1;
  v67 = 0;
  v68 = 0;
  v69 = 0;
  v64 = 0;
  v65 = 1;
  v62.m128i_i64[0] = 1;
  v87 = 0;
  sub_27A4170((__int64)&v67, &v86);
  sub_C8CF70((__int64)v79, v82, 8, (__int64)&v61[4], (__int64)v61);
  v5 = v61[12];
  memset(&v61[12], 0, 24);
  v83 = v5;
  v84 = v61[13];
  v85 = v61[14];
  sub_C8CF70((__int64)&v70, v75, 8, (__int64)v66, (__int64)&v62);
  v6 = v67;
  v67 = 0;
  v76 = v6;
  v7 = v68;
  v68 = 0;
  v77 = v7;
  v8 = v69;
  v69 = 0;
  v78 = v8;
  sub_C8CF70((__int64)&v86, v89, 8, (__int64)v75, (__int64)&v70);
  v9 = v76;
  v76 = 0;
  v90 = (const __m128i *)v9;
  v10 = v77;
  v77 = 0;
  v91 = v10;
  v11 = v78;
  v78 = 0;
  v92 = v11;
  sub_C8CF70((__int64)v93, v96, 8, (__int64)v82, (__int64)v79);
  v15 = v83;
  v83 = 0;
  v97 = (const __m128i *)v15;
  v16 = v84;
  v84 = 0;
  v98 = (const __m128i *)v16;
  v17 = v85;
  v85 = 0;
  v99 = v17;
  if ( v76 )
    j_j___libc_free_0(v76);
  if ( v74 )
  {
    v18 = v83;
    if ( !v83 )
      goto LABEL_8;
    goto LABEL_7;
  }
  _libc_free((unsigned __int64)v71);
  v18 = v83;
  if ( v83 )
LABEL_7:
    j_j___libc_free_0(v18);
LABEL_8:
  if ( v81 )
  {
    v19 = v67;
    if ( !v67 )
      goto LABEL_11;
    goto LABEL_10;
  }
  _libc_free(v80);
  v19 = v67;
  if ( v67 )
LABEL_10:
    j_j___libc_free_0(v19);
LABEL_11:
  if ( v65 )
  {
    v20 = v61[12];
    if ( !v61[12] )
      goto LABEL_14;
    goto LABEL_13;
  }
  _libc_free(v62.m128i_u64[1]);
  v20 = v61[12];
  if ( v61[12] )
LABEL_13:
    j_j___libc_free_0(v20);
LABEL_14:
  if ( !BYTE4(v61[3]) )
    _libc_free(v61[1]);
  v21 = (const __m128i *)v75;
  sub_C8CD80((__int64)&v70, (__int64)v75, (__int64)&v86, v12, v13, v14);
  v25 = v91;
  v26 = v90;
  v76 = 0;
  v77 = 0;
  v78 = 0;
  v27 = v91 - (_QWORD)v90;
  if ( (const __m128i *)v91 == v90 )
  {
    v27 = 0;
    v29 = 0;
  }
  else
  {
    if ( v27 > 0x7FFFFFFFFFFFFFF8LL )
      goto LABEL_99;
    v28 = sub_22077B0(v91 - (_QWORD)v90);
    v25 = v91;
    v26 = v90;
    v29 = v28;
  }
  v76 = v29;
  v77 = v29;
  v78 = v29 + v27;
  if ( (const __m128i *)v25 != v26 )
  {
    v30 = (__m128i *)v29;
    v31 = v26;
    do
    {
      if ( v30 )
      {
        *v30 = _mm_loadu_si128(v31);
        v23 = v31[1].m128i_i64[0];
        v30[1].m128i_i64[0] = v23;
      }
      v31 = (const __m128i *)((char *)v31 + 24);
      v30 = (__m128i *)((char *)v30 + 24);
    }
    while ( (const __m128i *)v25 != v31 );
    v29 += 8 * ((unsigned __int64)(v25 - 24 - (_QWORD)v26) >> 3) + 24;
  }
  v77 = v29;
  v26 = (const __m128i *)v79;
  sub_C8CD80((__int64)v79, (__int64)v82, (__int64)v93, v25, v23, v24);
  v34 = v98;
  v21 = v97;
  v83 = 0;
  v84 = 0;
  v85 = 0;
  v35 = (char *)v98 - (char *)v97;
  if ( v98 != v97 )
  {
    if ( v35 <= 0x7FFFFFFFFFFFFFF8LL )
    {
      v36 = sub_22077B0((char *)v98 - (char *)v97);
      v34 = v98;
      v21 = v97;
      v37 = v36;
      goto LABEL_28;
    }
LABEL_99:
    sub_4261EA(v26, v21, v22);
  }
  v37 = 0;
LABEL_28:
  v83 = v37;
  v38 = (__m128i *)v37;
  v84 = v37;
  v85 = v37 + v35;
  if ( v34 != v21 )
  {
    v39 = v21;
    do
    {
      if ( v38 )
      {
        *v38 = _mm_loadu_si128(v39);
        v32 = v39[1].m128i_i64[0];
        v38[1].m128i_i64[0] = v32;
      }
      v39 = (const __m128i *)((char *)v39 + 24);
      v38 = (__m128i *)((char *)v38 + 24);
    }
    while ( v34 != v39 );
    v38 = (__m128i *)(v37 + 8 * ((unsigned __int64)((char *)&v34[-2].m128i_u64[1] - (char *)v21) >> 3) + 24);
  }
  v84 = (unsigned __int64)v38;
  v40 = v76;
  v41 = v77;
LABEL_35:
  if ( (__m128i *)(v41 - v40) != (__m128i *)((char *)v38 - v37) )
  {
LABEL_36:
    v42 = *(__int64 **)(v41 - 24);
    v43 = *v42;
    if ( *v42 )
    {
      v44 = *v42;
      v62 = 0u;
      v63 = 0;
      v64 = 0;
      sub_27A7030(a1, v44, a2, (__int64)&v62);
      sub_27A2930(a1, v43, a3, (__int64)&v62);
      v45 = v64;
      if ( !v64 )
      {
LABEL_47:
        sub_C7D6A0(v62.m128i_i64[1], 48 * v45, 8);
        v41 = v77;
        goto LABEL_48;
      }
      v46 = v62.m128i_i64[1];
      v47 = v62.m128i_i64[1] + 48LL * v64;
      while ( *(_DWORD *)v46 == -1 )
      {
        if ( *(_QWORD *)(v46 + 8) == -1 )
        {
          v46 += 48;
          if ( v47 == v46 )
          {
LABEL_46:
            v45 = v64;
            goto LABEL_47;
          }
        }
        else
        {
LABEL_40:
          v48 = *(_QWORD *)(v46 + 16);
          if ( v48 != v46 + 32 )
            _libc_free(v48);
LABEL_42:
          v46 += 48;
          if ( v47 == v46 )
            goto LABEL_46;
        }
      }
      if ( *(_DWORD *)v46 == -2 && *(_QWORD *)(v46 + 8) == -2 )
        goto LABEL_42;
      goto LABEL_40;
    }
LABEL_48:
    v49 = *(_QWORD *)(v41 - 24);
    if ( !*(_BYTE *)(v41 - 8) )
    {
      v50 = *(__int64 **)(v49 + 24);
      *(_BYTE *)(v41 - 8) = 1;
      *(_QWORD *)(v41 - 16) = v50;
      goto LABEL_50;
    }
    while ( 1 )
    {
      while ( 1 )
      {
        v50 = *(__int64 **)(v41 - 16);
LABEL_50:
        v51 = *(unsigned int *)(v49 + 32);
        if ( v50 == (__int64 *)(*(_QWORD *)(v49 + 24) + 8 * v51) )
        {
          v77 -= 24LL;
          v40 = v76;
          v41 = v77;
          if ( v77 == v76 )
          {
LABEL_89:
            v37 = v83;
            v38 = (__m128i *)v84;
            goto LABEL_35;
          }
          goto LABEL_48;
        }
        v52 = v50 + 1;
        *(_QWORD *)(v41 - 16) = v50 + 1;
        v53 = *v50;
        if ( v74 )
          break;
LABEL_87:
        sub_C8CC70((__int64)&v70, v53, (__int64)v52, v51, v32, v33);
        if ( v57 )
          goto LABEL_88;
      }
      v54 = v71;
      v52 = &v71[v73];
      if ( v71 == v52 )
      {
LABEL_90:
        if ( v73 < v72 )
        {
          ++v73;
          *v52 = v53;
          ++v70;
LABEL_88:
          v62.m128i_i64[0] = v53;
          LOBYTE(v63) = 0;
          sub_27A4170((__int64)&v76, &v62);
          v40 = v76;
          v41 = v77;
          goto LABEL_89;
        }
        goto LABEL_87;
      }
      while ( v53 != *v54 )
      {
        if ( v52 == ++v54 )
          goto LABEL_90;
      }
    }
  }
  if ( v41 != v40 )
  {
    v55 = v37;
    while ( *(_QWORD *)v40 == *(_QWORD *)v55 )
    {
      v56 = *(_BYTE *)(v40 + 16);
      if ( v56 != *(_BYTE *)(v55 + 16) || v56 && *(_QWORD *)(v40 + 8) != *(_QWORD *)(v55 + 8) )
        break;
      v40 += 24LL;
      v55 += 24LL;
      if ( v41 == v40 )
        goto LABEL_63;
    }
    goto LABEL_36;
  }
LABEL_63:
  if ( v37 )
    j_j___libc_free_0(v37);
  if ( !v81 )
    _libc_free(v80);
  if ( v76 )
    j_j___libc_free_0(v76);
  if ( !v74 )
    _libc_free((unsigned __int64)v71);
  if ( v97 )
    j_j___libc_free_0((unsigned __int64)v97);
  if ( !v95 )
    _libc_free(v94);
  if ( v90 )
    j_j___libc_free_0((unsigned __int64)v90);
  if ( !v88 )
    _libc_free(v86.m128i_u64[1]);
}
