// Function: sub_E3ACE0
// Address: 0xe3ace0
//
__int64 __fastcall sub_E3ACE0(__int64 a1)
{
  __m128i *v1; // rax
  __int64 v2; // rax
  __int64 v3; // rax
  unsigned __int64 v4; // rax
  const __m128i *v5; // rax
  _BYTE *v6; // rsi
  __int64 v7; // rax
  unsigned __int64 v8; // rax
  __int64 v9; // rcx
  __int64 v10; // r8
  __int64 v11; // r9
  __m128i *v12; // rax
  __m128i *v13; // rax
  __int8 *v14; // rax
  __int64 v15; // rdx
  __int64 v16; // r8
  __int64 v17; // r9
  const __m128i *v18; // rsi
  const __m128i *v19; // rdi
  unsigned __int64 v20; // r14
  __int64 v21; // rax
  __int64 v22; // rcx
  __m128i *v23; // rdx
  const __m128i *v24; // rax
  __int64 v25; // r8
  __int64 v26; // r9
  const __m128i *v27; // rcx
  unsigned __int64 v28; // r13
  __int64 v29; // rax
  __m128i *v30; // rdi
  __m128i *v31; // rdx
  const __m128i *v32; // rax
  __int64 v33; // r14
  __int64 result; // rax
  __int64 v35; // rdx
  int v36; // eax
  __int64 v37; // rcx
  __int64 v38; // r15
  __int64 *v39; // rax
  __int64 *v40; // rdx
  __int64 v41; // r13
  __int64 *v42; // rax
  char v43; // dl
  __m128i *v44; // rdx
  char v45; // cl
  unsigned __int64 v46; // rsi
  _QWORD v47[16]; // [rsp+10h] [rbp-320h] BYREF
  __m128i v48; // [rsp+90h] [rbp-2A0h] BYREF
  __int64 v49; // [rsp+A0h] [rbp-290h]
  int v50; // [rsp+A8h] [rbp-288h]
  char v51; // [rsp+ACh] [rbp-284h]
  _QWORD v52[8]; // [rsp+B0h] [rbp-280h] BYREF
  __int64 v53; // [rsp+F0h] [rbp-240h] BYREF
  __int64 v54; // [rsp+F8h] [rbp-238h]
  unsigned __int64 v55; // [rsp+100h] [rbp-230h]
  __int64 v56; // [rsp+110h] [rbp-220h] BYREF
  __int64 *v57; // [rsp+118h] [rbp-218h]
  unsigned int v58; // [rsp+120h] [rbp-210h]
  unsigned int v59; // [rsp+124h] [rbp-20Ch]
  char v60; // [rsp+12Ch] [rbp-204h]
  _BYTE v61[64]; // [rsp+130h] [rbp-200h] BYREF
  __int64 v62; // [rsp+170h] [rbp-1C0h] BYREF
  __int64 v63; // [rsp+178h] [rbp-1B8h]
  unsigned __int64 v64; // [rsp+180h] [rbp-1B0h]
  char v65[8]; // [rsp+190h] [rbp-1A0h] BYREF
  __int64 v66; // [rsp+198h] [rbp-198h]
  char v67; // [rsp+1ACh] [rbp-184h]
  _BYTE v68[64]; // [rsp+1B0h] [rbp-180h] BYREF
  __m128i *v69; // [rsp+1F0h] [rbp-140h]
  __m128i *v70; // [rsp+1F8h] [rbp-138h]
  __int8 *v71; // [rsp+200h] [rbp-130h]
  __m128i v72; // [rsp+210h] [rbp-120h] BYREF
  char v73; // [rsp+220h] [rbp-110h]
  char v74; // [rsp+22Ch] [rbp-104h]
  char v75[64]; // [rsp+230h] [rbp-100h] BYREF
  const __m128i *v76; // [rsp+270h] [rbp-C0h]
  const __m128i *v77; // [rsp+278h] [rbp-B8h]
  unsigned __int64 v78; // [rsp+280h] [rbp-B0h]
  char v79[8]; // [rsp+288h] [rbp-A8h] BYREF
  __int64 v80; // [rsp+290h] [rbp-A0h]
  char v81; // [rsp+2A4h] [rbp-8Ch]
  _BYTE v82[64]; // [rsp+2A8h] [rbp-88h] BYREF
  const __m128i *v83; // [rsp+2E8h] [rbp-48h]
  const __m128i *v84; // [rsp+2F0h] [rbp-40h]
  __int8 *v85; // [rsp+2F8h] [rbp-38h]

  v48.m128i_i64[1] = (__int64)v52;
  memset(v47, 0, 0x78u);
  v47[1] = &v47[4];
  v49 = 0x100000008LL;
  v52[0] = a1;
  v72.m128i_i64[0] = a1;
  LODWORD(v47[2]) = 8;
  BYTE4(v47[3]) = 1;
  v53 = 0;
  v54 = 0;
  v55 = 0;
  v50 = 0;
  v51 = 1;
  v48.m128i_i64[0] = 1;
  v73 = 0;
  sub_E3ACA0((__int64)&v53, &v72);
  sub_C8CF70((__int64)v65, v68, 8, (__int64)&v47[4], (__int64)v47);
  v1 = (__m128i *)v47[12];
  memset(&v47[12], 0, 24);
  v69 = v1;
  v70 = (__m128i *)v47[13];
  v71 = (__int8 *)v47[14];
  sub_C8CF70((__int64)&v56, v61, 8, (__int64)v52, (__int64)&v48);
  v2 = v53;
  v53 = 0;
  v62 = v2;
  v3 = v54;
  v54 = 0;
  v63 = v3;
  v4 = v55;
  v55 = 0;
  v64 = v4;
  sub_C8CF70((__int64)&v72, v75, 8, (__int64)v61, (__int64)&v56);
  v5 = (const __m128i *)v62;
  v6 = v82;
  v62 = 0;
  v76 = v5;
  v7 = v63;
  v63 = 0;
  v77 = (const __m128i *)v7;
  v8 = v64;
  v64 = 0;
  v78 = v8;
  sub_C8CF70((__int64)v79, v82, 8, (__int64)v68, (__int64)v65);
  v12 = v69;
  v69 = 0;
  v83 = v12;
  v13 = v70;
  v70 = 0;
  v84 = v13;
  v14 = v71;
  v71 = 0;
  v85 = v14;
  if ( v62 )
  {
    v6 = (_BYTE *)(v64 - v62);
    j_j___libc_free_0(v62, v64 - v62);
  }
  if ( !v60 )
    _libc_free(v57, v6);
  if ( v69 )
  {
    v6 = (_BYTE *)(v71 - (__int8 *)v69);
    j_j___libc_free_0(v69, v71 - (__int8 *)v69);
  }
  if ( !v67 )
    _libc_free(v66, v6);
  if ( v53 )
  {
    v6 = (_BYTE *)(v55 - v53);
    j_j___libc_free_0(v53, v55 - v53);
  }
  if ( !v51 )
    _libc_free(v48.m128i_i64[1], v6);
  if ( v47[12] )
  {
    v6 = (_BYTE *)(v47[14] - v47[12]);
    j_j___libc_free_0(v47[12], v47[14] - v47[12]);
  }
  if ( !BYTE4(v47[3]) )
    _libc_free(v47[1], v6);
  sub_C8CD80((__int64)&v56, (__int64)v61, (__int64)&v72, v9, v10, v11);
  v18 = v77;
  v19 = v76;
  v62 = 0;
  v63 = 0;
  v64 = 0;
  v20 = (char *)v77 - (char *)v76;
  if ( v77 == v76 )
  {
    v22 = 0;
  }
  else
  {
    if ( v20 > 0x7FFFFFFFFFFFFFF8LL )
      goto LABEL_80;
    v21 = sub_22077B0((char *)v77 - (char *)v76);
    v18 = v77;
    v19 = v76;
    v22 = v21;
  }
  v62 = v22;
  v63 = v22;
  v64 = v22 + v20;
  if ( v18 != v19 )
  {
    v23 = (__m128i *)v22;
    v24 = v19;
    do
    {
      if ( v23 )
      {
        *v23 = _mm_loadu_si128(v24);
        v16 = v24[1].m128i_i64[0];
        v23[1].m128i_i64[0] = v16;
      }
      v24 = (const __m128i *)((char *)v24 + 24);
      v23 = (__m128i *)((char *)v23 + 24);
    }
    while ( v24 != v18 );
    v22 += 8 * ((unsigned __int64)((char *)&v24[-2].m128i_u64[1] - (char *)v19) >> 3) + 24;
  }
  v19 = (const __m128i *)v65;
  v63 = v22;
  sub_C8CD80((__int64)v65, (__int64)v68, (__int64)v79, v22, v16, v17);
  v27 = v84;
  v18 = v83;
  v69 = 0;
  v70 = 0;
  v71 = 0;
  v28 = (char *)v84 - (char *)v83;
  if ( v84 != v83 )
  {
    if ( v28 <= 0x7FFFFFFFFFFFFFF8LL )
    {
      v29 = sub_22077B0((char *)v84 - (char *)v83);
      v27 = v84;
      v18 = v83;
      v30 = (__m128i *)v29;
      goto LABEL_29;
    }
LABEL_80:
    sub_4261EA(v19, v18, v15);
  }
  v30 = 0;
LABEL_29:
  v69 = v30;
  v31 = v30;
  v70 = v30;
  v71 = &v30->m128i_i8[v28];
  if ( v27 != v18 )
  {
    v32 = v18;
    do
    {
      if ( v31 )
      {
        *v31 = _mm_loadu_si128(v32);
        v25 = v32[1].m128i_i64[0];
        v31[1].m128i_i64[0] = v25;
      }
      v32 = (const __m128i *)((char *)v32 + 24);
      v31 = (__m128i *)((char *)v31 + 24);
    }
    while ( v27 != v32 );
    v31 = (__m128i *)((char *)v30 + 8 * ((unsigned __int64)((char *)&v27[-2].m128i_u64[1] - (char *)v18) >> 3) + 24);
  }
  v33 = v63;
  result = v62;
  v70 = v31;
  if ( v63 - v62 != (char *)v31 - (char *)v30 )
    goto LABEL_36;
LABEL_52:
  if ( result != v33 )
  {
    v44 = v30;
    while ( *(_QWORD *)result == v44->m128i_i64[0] )
    {
      v45 = *(_BYTE *)(result + 16);
      if ( v45 != v44[1].m128i_i8[0] || v45 && *(_QWORD *)(result + 8) != v44->m128i_i64[1] )
        break;
      result += 24;
      v44 = (__m128i *)((char *)v44 + 24);
      if ( result == v33 )
        goto LABEL_59;
    }
LABEL_36:
    v35 = *(_QWORD *)(v33 - 24);
    v36 = 1;
    v37 = *(_QWORD *)v35;
    if ( *(_QWORD *)v35 )
      v36 = *(_DWORD *)(v37 + 168) + 1;
    *(_DWORD *)(v35 + 168) = v36;
    while ( 1 )
    {
      v38 = *(_QWORD *)(v33 - 24);
      if ( *(_BYTE *)(v33 - 8) )
        break;
      v39 = *(__int64 **)(v38 + 32);
      *(_BYTE *)(v33 - 8) = 1;
      *(_QWORD *)(v33 - 16) = v39;
      if ( v39 != *(__int64 **)(v38 + 40) )
        goto LABEL_41;
LABEL_47:
      v63 -= 24;
      result = v62;
      v33 = v63;
      if ( v63 == v62 )
        goto LABEL_51;
    }
    while ( 1 )
    {
      v39 = *(__int64 **)(v33 - 16);
      if ( v39 == *(__int64 **)(v38 + 40) )
        goto LABEL_47;
LABEL_41:
      v40 = v39 + 1;
      *(_QWORD *)(v33 - 16) = v39 + 1;
      v41 = *v39;
      if ( !v60 )
        goto LABEL_49;
      v42 = v57;
      v37 = v59;
      v40 = &v57[v59];
      if ( v57 == v40 )
      {
LABEL_75:
        if ( v59 < v58 )
        {
          ++v59;
          *v40 = v41;
          ++v56;
LABEL_50:
          v48.m128i_i64[0] = v41;
          LOBYTE(v49) = 0;
          sub_E3ACA0((__int64)&v62, &v48);
          result = v62;
          v33 = v63;
LABEL_51:
          v30 = v69;
          if ( v33 - result == (char *)v70 - (char *)v69 )
            goto LABEL_52;
          goto LABEL_36;
        }
LABEL_49:
        sub_C8CC70((__int64)&v56, v41, (__int64)v40, v37, v25, v26);
        if ( v43 )
          goto LABEL_50;
      }
      else
      {
        while ( v41 != *v42 )
        {
          if ( v40 == ++v42 )
            goto LABEL_75;
        }
      }
    }
  }
LABEL_59:
  v46 = v71 - (__int8 *)v30;
  if ( v30 )
    result = j_j___libc_free_0(v30, v46);
  if ( !v67 )
    result = _libc_free(v66, v46);
  if ( v62 )
  {
    v46 = v64 - v62;
    result = j_j___libc_free_0(v62, v64 - v62);
  }
  if ( !v60 )
    result = _libc_free(v57, v46);
  if ( v83 )
  {
    v46 = v85 - (__int8 *)v83;
    result = j_j___libc_free_0(v83, v85 - (__int8 *)v83);
  }
  if ( !v81 )
    result = _libc_free(v80, v46);
  if ( v76 )
  {
    v46 = v78 - (_QWORD)v76;
    result = j_j___libc_free_0(v76, v78 - (_QWORD)v76);
  }
  if ( !v74 )
    return _libc_free(v72.m128i_i64[1], v46);
  return result;
}
