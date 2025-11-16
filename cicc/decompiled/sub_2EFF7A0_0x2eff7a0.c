// Function: sub_2EFF7A0
// Address: 0x2eff7a0
//
__int64 __fastcall sub_2EFF7A0(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  __int64 v7; // rdx
  __int64 v8; // rcx
  __int64 v9; // r8
  __int64 v10; // r9
  __int64 v11; // rcx
  __int64 v12; // r8
  __int64 v13; // r9
  __int64 v14; // rdx
  __m128i *v15; // rcx
  __int64 v16; // r8
  __int64 v17; // r9
  unsigned int v18; // r13d
  __int64 v19; // rdx
  __m128i *v20; // rcx
  __int64 v21; // r8
  __int64 v22; // r9
  unsigned int v23; // ebx
  __int64 v24; // rdx
  __m128i *v25; // rcx
  __int64 v26; // r8
  __int64 v27; // r9
  int v28; // ebx
  __int64 v29; // r9
  int v30; // edx
  _BYTE *v31; // rdi
  __int64 v32; // rcx
  __int64 v33; // rsi
  __int64 v34; // rdx
  _QWORD *v35; // rcx
  __int64 v36; // r9
  __int64 v37; // rdx
  __int64 v38; // r8
  __int64 v39; // rcx
  __int64 v41; // rsi
  const __m128i *v42; // rdx
  const __m128i *v43; // rdi
  __int64 v44; // rsi
  const __m128i *v45; // rdx
  const __m128i *v46; // rdi
  __int64 v47; // r8
  const __m128i *v48; // rsi
  const __m128i *v49; // r9
  __int64 v50; // rsi
  const __m128i *v51; // rdx
  const __m128i *v52; // rdi
  __int64 v53; // [rsp+8h] [rbp-9C8h]
  int v54; // [rsp+8h] [rbp-9C8h]
  __int64 v55; // [rsp+10h] [rbp-9C0h]
  _BYTE v56[8]; // [rsp+20h] [rbp-9B0h] BYREF
  unsigned __int64 v57; // [rsp+28h] [rbp-9A8h]
  char v58; // [rsp+3Ch] [rbp-994h]
  char *v59; // [rsp+80h] [rbp-950h]
  char v60; // [rsp+90h] [rbp-940h] BYREF
  _BYTE v61[8]; // [rsp+150h] [rbp-880h] BYREF
  unsigned __int64 v62; // [rsp+158h] [rbp-878h]
  char v63; // [rsp+16Ch] [rbp-864h]
  const __m128i *v64; // [rsp+1B0h] [rbp-820h]
  unsigned int v65; // [rsp+1B8h] [rbp-818h]
  char v66; // [rsp+1C0h] [rbp-810h] BYREF
  _BYTE v67[8]; // [rsp+280h] [rbp-750h] BYREF
  unsigned __int64 v68; // [rsp+288h] [rbp-748h]
  char v69; // [rsp+29Ch] [rbp-734h]
  char *v70; // [rsp+2E0h] [rbp-6F0h]
  char v71; // [rsp+2F0h] [rbp-6E0h] BYREF
  _BYTE v72[8]; // [rsp+3B0h] [rbp-620h] BYREF
  unsigned __int64 v73; // [rsp+3B8h] [rbp-618h]
  char v74; // [rsp+3CCh] [rbp-604h]
  const __m128i *v75; // [rsp+410h] [rbp-5C0h]
  unsigned int v76; // [rsp+418h] [rbp-5B8h]
  char v77; // [rsp+420h] [rbp-5B0h] BYREF
  _BYTE v78[8]; // [rsp+4E0h] [rbp-4F0h] BYREF
  unsigned __int64 v79; // [rsp+4E8h] [rbp-4E8h]
  char v80; // [rsp+4FCh] [rbp-4D4h]
  _BYTE v81[64]; // [rsp+500h] [rbp-4D0h] BYREF
  __m128i *v82; // [rsp+540h] [rbp-490h] BYREF
  __int64 v83; // [rsp+548h] [rbp-488h]
  _BYTE v84[192]; // [rsp+550h] [rbp-480h] BYREF
  _BYTE v85[8]; // [rsp+610h] [rbp-3C0h] BYREF
  unsigned __int64 v86; // [rsp+618h] [rbp-3B8h]
  char v87; // [rsp+62Ch] [rbp-3A4h]
  _BYTE v88[64]; // [rsp+630h] [rbp-3A0h] BYREF
  __m128i *v89; // [rsp+670h] [rbp-360h] BYREF
  __int64 v90; // [rsp+678h] [rbp-358h]
  _BYTE v91[192]; // [rsp+680h] [rbp-350h] BYREF
  _BYTE v92[8]; // [rsp+740h] [rbp-290h] BYREF
  unsigned __int64 v93; // [rsp+748h] [rbp-288h]
  char v94; // [rsp+75Ch] [rbp-274h]
  _BYTE v95[64]; // [rsp+760h] [rbp-270h] BYREF
  _BYTE *v96; // [rsp+7A0h] [rbp-230h] BYREF
  __int64 v97; // [rsp+7A8h] [rbp-228h]
  _BYTE v98[192]; // [rsp+7B0h] [rbp-220h] BYREF
  _BYTE v99[8]; // [rsp+870h] [rbp-160h] BYREF
  unsigned __int64 v100; // [rsp+878h] [rbp-158h]
  char v101; // [rsp+88Ch] [rbp-144h]
  _BYTE v102[64]; // [rsp+890h] [rbp-140h] BYREF
  __m128i *v103; // [rsp+8D0h] [rbp-100h] BYREF
  __int64 v104; // [rsp+8D8h] [rbp-F8h]
  _BYTE v105[240]; // [rsp+8E0h] [rbp-F0h] BYREF

  sub_2EFF6E0((__int64)v67, a2, a3, a4, a5, a6);
  sub_2EFF5C0((__int64)v72, (__int64)v67);
  sub_2EFF6E0((__int64)v56, a1, v7, v8, v9, v10);
  sub_2EFF5C0((__int64)v61, (__int64)v56);
  sub_C8CD80((__int64)v85, (__int64)v88, (__int64)v72, v11, v12, v13);
  v18 = v76;
  v89 = (__m128i *)v91;
  v90 = 0x800000000LL;
  if ( v76 )
  {
    v15 = (__m128i *)v91;
    v41 = v76;
    if ( v76 > 8 )
    {
      sub_2E3C030((__int64)&v89, v76, v14, (__int64)v91, v16, v17);
      v15 = v89;
      v41 = v76;
    }
    v42 = v75;
    v43 = (const __m128i *)((char *)v75 + 24 * v41);
    if ( v75 != v43 )
    {
      do
      {
        if ( v15 )
        {
          *v15 = _mm_loadu_si128(v42);
          v15[1].m128i_i64[0] = v42[1].m128i_i64[0];
        }
        v42 = (const __m128i *)((char *)v42 + 24);
        v15 = (__m128i *)((char *)v15 + 24);
      }
      while ( v43 != v42 );
    }
    LODWORD(v90) = v18;
  }
  sub_C8CD80((__int64)v78, (__int64)v81, (__int64)v61, (__int64)v15, v16, v17);
  v23 = v65;
  v82 = (__m128i *)v84;
  v83 = 0x800000000LL;
  if ( v65 )
  {
    v50 = v65;
    v20 = (__m128i *)v84;
    if ( v65 > 8 )
    {
      sub_2E3C030((__int64)&v82, v65, v19, (__int64)v84, v21, v22);
      v20 = v82;
      v50 = v65;
    }
    v51 = v64;
    v52 = (const __m128i *)((char *)v64 + 24 * v50);
    if ( v64 != v52 )
    {
      do
      {
        if ( v20 )
        {
          *v20 = _mm_loadu_si128(v51);
          v20[1].m128i_i64[0] = v51[1].m128i_i64[0];
        }
        v51 = (const __m128i *)((char *)v51 + 24);
        v20 = (__m128i *)((char *)v20 + 24);
      }
      while ( v52 != v51 );
    }
    LODWORD(v83) = v23;
  }
  sub_C8CD80((__int64)v99, (__int64)v102, (__int64)v85, (__int64)v20, v21, v22);
  v28 = v90;
  v103 = (__m128i *)v105;
  v104 = 0x800000000LL;
  if ( (_DWORD)v90 )
  {
    v44 = (unsigned int)v90;
    v25 = (__m128i *)v105;
    if ( (unsigned int)v90 > 8 )
    {
      sub_2E3C030((__int64)&v103, (unsigned int)v90, v24, (__int64)v105, v26, v27);
      v25 = v103;
      v44 = (unsigned int)v90;
    }
    v45 = v89;
    v46 = (__m128i *)((char *)v89 + 24 * v44);
    if ( v89 != v46 )
    {
      do
      {
        if ( v25 )
        {
          *v25 = _mm_loadu_si128(v45);
          v25[1].m128i_i64[0] = v45[1].m128i_i64[0];
        }
        v45 = (const __m128i *)((char *)v45 + 24);
        v25 = (__m128i *)((char *)v25 + 24);
      }
      while ( v46 != v45 );
    }
    LODWORD(v104) = v28;
  }
  sub_C8CD80((__int64)v92, (__int64)v95, (__int64)v78, (__int64)v25, v26, v27);
  v30 = v83;
  v96 = v98;
  v97 = 0x800000000LL;
  if ( (_DWORD)v83 )
  {
    v32 = (unsigned int)v83;
    v31 = v98;
    v47 = (unsigned int)v83;
    if ( (unsigned int)v83 > 8 )
    {
      v54 = v83;
      v55 = (unsigned int)v83;
      sub_2E3C030((__int64)&v96, (unsigned int)v83, (unsigned int)v83, (unsigned int)v83, (unsigned int)v83, v29);
      v31 = v96;
      v47 = (unsigned int)v83;
      v30 = v54;
      v32 = v55;
    }
    v48 = v82;
    v49 = (__m128i *)((char *)v82 + 24 * v47);
    if ( v82 != v49 )
    {
      do
      {
        if ( v31 )
        {
          *(__m128i *)v31 = _mm_loadu_si128(v48);
          *((_QWORD *)v31 + 2) = v48[1].m128i_i64[0];
        }
        v48 = (const __m128i *)((char *)v48 + 24);
        v31 += 24;
      }
      while ( v49 != v48 );
      v31 = v96;
    }
    LODWORD(v97) = v30;
  }
  else
  {
    v31 = v98;
    v32 = 0;
  }
  while ( 1 )
  {
    v33 = (unsigned int)v104;
    v34 = 24 * v32;
    if ( v32 != (unsigned int)v104 )
      goto LABEL_10;
    v33 = (__int64)v103;
    if ( v31 == &v31[v34] )
      break;
    v35 = v31;
    while ( v35[2] == *(_QWORD *)(v33 + 16) && v35[1] == *(_QWORD *)(v33 + 8) && *v35 == *(_QWORD *)v33 )
    {
      v35 += 3;
      v33 += 24;
      if ( &v31[v34] == (_BYTE *)v35 )
        goto LABEL_18;
    }
LABEL_10:
    v36 = *(_QWORD *)&v31[v34 - 8];
    v37 = *(unsigned int *)(a3 + 8);
    v38 = v37 + 1;
    if ( v37 + 1 > (unsigned __int64)*(unsigned int *)(a3 + 12) )
    {
      v33 = a3 + 16;
      v53 = v36;
      sub_C8D5F0(a3, (const void *)(a3 + 16), v37 + 1, 8u, v38, v36);
      v37 = *(unsigned int *)(a3 + 8);
      v36 = v53;
    }
    v39 = *(_QWORD *)a3;
    *(_QWORD *)(*(_QWORD *)a3 + 8 * v37) = v36;
    ++*(_DWORD *)(a3 + 8);
    LODWORD(v97) = v97 - 1;
    if ( (_DWORD)v97 )
    {
      sub_2EFF3E0((__int64)v92, v33, v37, v39, v38, v36);
      v31 = v96;
      v32 = (unsigned int)v97;
    }
    else
    {
      v31 = v96;
      v32 = 0;
    }
  }
LABEL_18:
  if ( v31 != v98 )
    _libc_free((unsigned __int64)v31);
  if ( !v94 )
    _libc_free(v93);
  if ( v103 != (__m128i *)v105 )
    _libc_free((unsigned __int64)v103);
  if ( !v101 )
    _libc_free(v100);
  if ( v82 != (__m128i *)v84 )
    _libc_free((unsigned __int64)v82);
  if ( !v80 )
    _libc_free(v79);
  if ( v89 != (__m128i *)v91 )
    _libc_free((unsigned __int64)v89);
  if ( !v87 )
    _libc_free(v86);
  if ( v64 != (const __m128i *)&v66 )
    _libc_free((unsigned __int64)v64);
  if ( !v63 )
    _libc_free(v62);
  if ( v59 != &v60 )
    _libc_free((unsigned __int64)v59);
  if ( !v58 )
    _libc_free(v57);
  if ( v75 != (const __m128i *)&v77 )
    _libc_free((unsigned __int64)v75);
  if ( !v74 )
    _libc_free(v73);
  if ( v70 != &v71 )
    _libc_free((unsigned __int64)v70);
  if ( !v69 )
    _libc_free(v68);
  return a3;
}
