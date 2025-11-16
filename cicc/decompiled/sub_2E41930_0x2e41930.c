// Function: sub_2E41930
// Address: 0x2e41930
//
__int64 __fastcall sub_2E41930(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  __int64 v7; // rdx
  __int64 v8; // rcx
  __int64 v9; // r8
  __int64 v10; // r9
  __int64 v11; // rcx
  __int64 v12; // r8
  __int64 v13; // r9
  __int64 v14; // rcx
  __int64 v15; // r8
  __int64 v16; // r9
  unsigned int v17; // r13d
  __int64 v18; // rcx
  __int64 v19; // r8
  __int64 v20; // r9
  unsigned int v21; // ebx
  __int64 v22; // rcx
  __int64 v23; // r8
  __int64 v24; // r9
  int v25; // ebx
  __int64 v26; // rdx
  __int64 v27; // r8
  __int64 v28; // r9
  int v29; // eax
  _BYTE *v30; // rdi
  __int64 v31; // rcx
  __int64 v32; // rdx
  __int64 *v33; // rsi
  _BYTE *v34; // rsi
  __int64 v35; // rdx
  __m128i *v37; // rdx
  __int64 v38; // rsi
  const __m128i *v39; // rax
  const __m128i *v40; // rsi
  __int64 v41; // rsi
  __m128i *v42; // rdx
  const __m128i *v43; // rax
  const __m128i *v44; // rsi
  __int64 v45; // rsi
  const __m128i *v46; // rdx
  __int64 v47; // rsi
  __m128i *v48; // rdx
  const __m128i *v49; // rax
  const __m128i *v50; // rsi
  int v51; // [rsp+Ch] [rbp-9C4h]
  __int64 v52; // [rsp+10h] [rbp-9C0h]
  char v53[8]; // [rsp+20h] [rbp-9B0h] BYREF
  unsigned __int64 v54; // [rsp+28h] [rbp-9A8h]
  char v55; // [rsp+3Ch] [rbp-994h]
  char *v56; // [rsp+80h] [rbp-950h]
  char v57; // [rsp+90h] [rbp-940h] BYREF
  char v58[8]; // [rsp+150h] [rbp-880h] BYREF
  unsigned __int64 v59; // [rsp+158h] [rbp-878h]
  char v60; // [rsp+16Ch] [rbp-864h]
  const __m128i *v61; // [rsp+1B0h] [rbp-820h]
  unsigned int v62; // [rsp+1B8h] [rbp-818h]
  char v63; // [rsp+1C0h] [rbp-810h] BYREF
  char v64[8]; // [rsp+280h] [rbp-750h] BYREF
  unsigned __int64 v65; // [rsp+288h] [rbp-748h]
  char v66; // [rsp+29Ch] [rbp-734h]
  char *v67; // [rsp+2E0h] [rbp-6F0h]
  char v68; // [rsp+2F0h] [rbp-6E0h] BYREF
  char v69[8]; // [rsp+3B0h] [rbp-620h] BYREF
  unsigned __int64 v70; // [rsp+3B8h] [rbp-618h]
  char v71; // [rsp+3CCh] [rbp-604h]
  const __m128i *v72; // [rsp+410h] [rbp-5C0h]
  unsigned int v73; // [rsp+418h] [rbp-5B8h]
  char v74; // [rsp+420h] [rbp-5B0h] BYREF
  char v75[8]; // [rsp+4E0h] [rbp-4F0h] BYREF
  unsigned __int64 v76; // [rsp+4E8h] [rbp-4E8h]
  char v77; // [rsp+4FCh] [rbp-4D4h]
  char v78[64]; // [rsp+500h] [rbp-4D0h] BYREF
  __m128i *v79; // [rsp+540h] [rbp-490h] BYREF
  __int64 v80; // [rsp+548h] [rbp-488h]
  _BYTE v81[192]; // [rsp+550h] [rbp-480h] BYREF
  char v82[8]; // [rsp+610h] [rbp-3C0h] BYREF
  unsigned __int64 v83; // [rsp+618h] [rbp-3B8h]
  char v84; // [rsp+62Ch] [rbp-3A4h]
  char v85[64]; // [rsp+630h] [rbp-3A0h] BYREF
  __m128i *v86; // [rsp+670h] [rbp-360h] BYREF
  __int64 v87; // [rsp+678h] [rbp-358h]
  _BYTE v88[192]; // [rsp+680h] [rbp-350h] BYREF
  char v89[8]; // [rsp+740h] [rbp-290h] BYREF
  unsigned __int64 v90; // [rsp+748h] [rbp-288h]
  char v91; // [rsp+75Ch] [rbp-274h]
  char v92[64]; // [rsp+760h] [rbp-270h] BYREF
  _BYTE *v93; // [rsp+7A0h] [rbp-230h] BYREF
  __int64 v94; // [rsp+7A8h] [rbp-228h]
  _BYTE v95[192]; // [rsp+7B0h] [rbp-220h] BYREF
  char v96[8]; // [rsp+870h] [rbp-160h] BYREF
  unsigned __int64 v97; // [rsp+878h] [rbp-158h]
  char v98; // [rsp+88Ch] [rbp-144h]
  char v99[64]; // [rsp+890h] [rbp-140h] BYREF
  __m128i *v100; // [rsp+8D0h] [rbp-100h] BYREF
  __int64 v101; // [rsp+8D8h] [rbp-F8h]
  _BYTE v102[240]; // [rsp+8E0h] [rbp-F0h] BYREF

  sub_2E3C1F0((__int64)v64, a2, a3, a4, a5, a6);
  sub_2E3C0D0((__int64)v69, (__int64)v64);
  sub_2E3C1F0((__int64)v53, a1, v7, v8, v9, v10);
  sub_2E3C0D0((__int64)v58, (__int64)v53);
  sub_C8CD80((__int64)v82, (__int64)v85, (__int64)v69, v11, v12, v13);
  v17 = v73;
  v86 = (__m128i *)v88;
  v87 = 0x800000000LL;
  if ( v73 )
  {
    v37 = (__m128i *)v88;
    v38 = v73;
    if ( v73 > 8 )
    {
      sub_2E3C030((__int64)&v86, v73, (__int64)v88, v14, v15, v16);
      v37 = v86;
      v38 = v73;
    }
    v39 = v72;
    v14 = 3 * v38;
    v40 = (const __m128i *)((char *)v72 + 24 * v38);
    if ( v72 != v40 )
    {
      do
      {
        if ( v37 )
        {
          *v37 = _mm_loadu_si128(v39);
          v14 = v39[1].m128i_i64[0];
          v37[1].m128i_i64[0] = v14;
        }
        v39 = (const __m128i *)((char *)v39 + 24);
        v37 = (__m128i *)((char *)v37 + 24);
      }
      while ( v40 != v39 );
    }
    LODWORD(v87) = v17;
  }
  sub_C8CD80((__int64)v75, (__int64)v78, (__int64)v58, v14, v15, v16);
  v21 = v62;
  v79 = (__m128i *)v81;
  v80 = 0x800000000LL;
  if ( v62 )
  {
    v47 = v62;
    v48 = (__m128i *)v81;
    if ( v62 > 8 )
    {
      sub_2E3C030((__int64)&v79, v62, (__int64)v81, v18, v19, v20);
      v48 = v79;
      v47 = v62;
    }
    v49 = v61;
    v18 = 3 * v47;
    v50 = (const __m128i *)((char *)v61 + 24 * v47);
    if ( v61 != v50 )
    {
      do
      {
        if ( v48 )
        {
          *v48 = _mm_loadu_si128(v49);
          v18 = v49[1].m128i_i64[0];
          v48[1].m128i_i64[0] = v18;
        }
        v49 = (const __m128i *)((char *)v49 + 24);
        v48 = (__m128i *)((char *)v48 + 24);
      }
      while ( v50 != v49 );
    }
    LODWORD(v80) = v21;
  }
  sub_C8CD80((__int64)v96, (__int64)v99, (__int64)v82, v18, v19, v20);
  v25 = v87;
  v100 = (__m128i *)v102;
  v101 = 0x800000000LL;
  if ( (_DWORD)v87 )
  {
    v41 = (unsigned int)v87;
    v42 = (__m128i *)v102;
    if ( (unsigned int)v87 > 8 )
    {
      sub_2E3C030((__int64)&v100, (unsigned int)v87, (__int64)v102, v22, v23, v24);
      v42 = v100;
      v41 = (unsigned int)v87;
    }
    v43 = v86;
    v22 = 3 * v41;
    v44 = (__m128i *)((char *)v86 + 24 * v41);
    if ( v86 != v44 )
    {
      do
      {
        if ( v42 )
        {
          *v42 = _mm_loadu_si128(v43);
          v22 = v43[1].m128i_i64[0];
          v42[1].m128i_i64[0] = v22;
        }
        v43 = (const __m128i *)((char *)v43 + 24);
        v42 = (__m128i *)((char *)v42 + 24);
      }
      while ( v44 != v43 );
    }
    LODWORD(v101) = v25;
  }
  sub_C8CD80((__int64)v89, (__int64)v92, (__int64)v75, v22, v23, v24);
  v93 = v95;
  v94 = 0x800000000LL;
  v29 = v80;
  if ( (_DWORD)v80 )
  {
    v31 = (unsigned int)v80;
    v30 = v95;
    v45 = (unsigned int)v80;
    if ( (unsigned int)v80 > 8 )
    {
      v51 = v80;
      v52 = (unsigned int)v80;
      sub_2E3C030((__int64)&v93, (unsigned int)v80, v26, (unsigned int)v80, v27, v28);
      v30 = v93;
      v45 = (unsigned int)v80;
      v29 = v51;
      v31 = v52;
    }
    v46 = v79;
    v28 = (__int64)&v79->m128i_i64[3 * v45];
    if ( v79 != (__m128i *)v28 )
    {
      do
      {
        if ( v30 )
        {
          *(__m128i *)v30 = _mm_loadu_si128(v46);
          *((_QWORD *)v30 + 2) = v46[1].m128i_i64[0];
        }
        v46 = (const __m128i *)((char *)v46 + 24);
        v30 += 24;
      }
      while ( (const __m128i *)v28 != v46 );
      v30 = v93;
    }
    LODWORD(v94) = v29;
  }
  else
  {
    v30 = v95;
    v31 = 0;
  }
  while ( 1 )
  {
    v32 = 24 * v31;
    if ( v31 != (unsigned int)v101 )
      goto LABEL_10;
    v28 = (__int64)&v30[v32];
    v33 = (__int64 *)v100;
    if ( v30 == &v30[v32] )
      break;
    v31 = (__int64)v30;
    while ( 1 )
    {
      v27 = v33[2];
      if ( *(_QWORD *)(v31 + 16) != v27 || *(_QWORD *)(v31 + 8) != v33[1] || *(_QWORD *)v31 != *v33 )
        break;
      v31 += 24;
      v33 += 3;
      if ( v28 == v31 )
        goto LABEL_20;
    }
LABEL_10:
    v34 = *(_BYTE **)(a3 + 8);
    v35 = (__int64)&v30[v32 - 24];
    if ( v34 == *(_BYTE **)(a3 + 16) )
    {
      sub_2E417A0(a3, v34, (_QWORD *)(v35 + 16));
      v29 = v94;
    }
    else
    {
      if ( v34 )
      {
        *(_QWORD *)v34 = *(_QWORD *)(v35 + 16);
        v34 = *(_BYTE **)(a3 + 8);
        v29 = v94;
      }
      v34 += 8;
      *(_QWORD *)(a3 + 8) = v34;
    }
    LODWORD(v94) = --v29;
    if ( v29 )
    {
      sub_2E3BE50((__int64)v89, (__int64)v34, v35, v31, v27, v28);
      v31 = (unsigned int)v94;
      v30 = v93;
      v29 = v94;
    }
    else
    {
      v30 = v93;
      v31 = 0;
    }
  }
LABEL_20:
  if ( v30 != v95 )
    _libc_free((unsigned __int64)v30);
  if ( !v91 )
    _libc_free(v90);
  if ( v100 != (__m128i *)v102 )
    _libc_free((unsigned __int64)v100);
  if ( !v98 )
    _libc_free(v97);
  if ( v79 != (__m128i *)v81 )
    _libc_free((unsigned __int64)v79);
  if ( !v77 )
    _libc_free(v76);
  if ( v86 != (__m128i *)v88 )
    _libc_free((unsigned __int64)v86);
  if ( !v84 )
    _libc_free(v83);
  if ( v61 != (const __m128i *)&v63 )
    _libc_free((unsigned __int64)v61);
  if ( !v60 )
    _libc_free(v59);
  if ( v56 != &v57 )
    _libc_free((unsigned __int64)v56);
  if ( !v55 )
    _libc_free(v54);
  if ( v72 != (const __m128i *)&v74 )
    _libc_free((unsigned __int64)v72);
  if ( !v71 )
    _libc_free(v70);
  if ( v67 != &v68 )
    _libc_free((unsigned __int64)v67);
  if ( !v66 )
    _libc_free(v65);
  return a3;
}
