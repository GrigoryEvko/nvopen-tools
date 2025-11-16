// Function: sub_2E564A0
// Address: 0x2e564a0
//
__int64 __fastcall sub_2E564A0(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  __int64 v7; // rdx
  __int64 v8; // rcx
  __int64 v9; // r8
  __int64 v10; // r9
  __int64 v11; // rcx
  __int64 v12; // r8
  __int64 v13; // r9
  const __m128i *v14; // rcx
  __int64 v15; // r8
  __int64 v16; // r9
  unsigned int v17; // r12d
  __m128i *v18; // rdx
  const __m128i *v19; // rcx
  __int64 v20; // r8
  __int64 v21; // r9
  unsigned int v22; // r12d
  __m128i *v23; // rdx
  const __m128i *v24; // rcx
  __int64 v25; // r8
  __int64 v26; // r9
  int v27; // r12d
  __m128i *v28; // rdx
  __int64 v29; // rdx
  __int64 v30; // r9
  int v31; // r8d
  __m128i *v32; // rcx
  __int64 v33; // rsi
  __int64 v34; // rax
  __int64 *v35; // rdx
  __int64 v36; // r8
  __int64 v37; // rax
  unsigned __int64 v38; // rcx
  __int64 v39; // rdx
  __int64 v41; // rsi
  const __m128i *v42; // rdi
  __int64 v43; // rsi
  const __m128i *v44; // rdi
  __int64 v45; // rdi
  const __m128i *v46; // rdx
  __int64 v47; // rsi
  const __m128i *v48; // rdi
  __int64 v49; // [rsp+8h] [rbp-9D8h]
  int v50; // [rsp+8h] [rbp-9D8h]
  __int64 v51; // [rsp+10h] [rbp-9D0h]
  _BYTE v52[304]; // [rsp+30h] [rbp-9B0h] BYREF
  _BYTE v53[96]; // [rsp+160h] [rbp-880h] BYREF
  const __m128i *v54; // [rsp+1C0h] [rbp-820h]
  unsigned int v55; // [rsp+1C8h] [rbp-818h]
  _BYTE v56[304]; // [rsp+290h] [rbp-750h] BYREF
  _BYTE v57[96]; // [rsp+3C0h] [rbp-620h] BYREF
  const __m128i *v58; // [rsp+420h] [rbp-5C0h]
  unsigned int v59; // [rsp+428h] [rbp-5B8h]
  _BYTE v60[32]; // [rsp+4F0h] [rbp-4F0h] BYREF
  _BYTE v61[64]; // [rsp+510h] [rbp-4D0h] BYREF
  __m128i *v62; // [rsp+550h] [rbp-490h] BYREF
  __int64 v63; // [rsp+558h] [rbp-488h]
  _BYTE v64[192]; // [rsp+560h] [rbp-480h] BYREF
  _BYTE v65[32]; // [rsp+620h] [rbp-3C0h] BYREF
  _BYTE v66[64]; // [rsp+640h] [rbp-3A0h] BYREF
  __m128i *v67; // [rsp+680h] [rbp-360h] BYREF
  __int64 v68; // [rsp+688h] [rbp-358h]
  _BYTE v69[192]; // [rsp+690h] [rbp-350h] BYREF
  _BYTE v70[32]; // [rsp+750h] [rbp-290h] BYREF
  _BYTE v71[64]; // [rsp+770h] [rbp-270h] BYREF
  __m128i *v72; // [rsp+7B0h] [rbp-230h] BYREF
  __int64 v73; // [rsp+7B8h] [rbp-228h]
  _BYTE v74[192]; // [rsp+7C0h] [rbp-220h] BYREF
  _BYTE v75[32]; // [rsp+880h] [rbp-160h] BYREF
  _BYTE v76[64]; // [rsp+8A0h] [rbp-140h] BYREF
  __m128i *v77; // [rsp+8E0h] [rbp-100h] BYREF
  __int64 v78; // [rsp+8E8h] [rbp-F8h]
  _BYTE v79[240]; // [rsp+8F0h] [rbp-F0h] BYREF

  sub_2E563E0((__int64)v56, a2, a3, a4, a5, a6);
  sub_2DACDE0((__int64)v57, (__int64)v56);
  sub_2E563E0((__int64)v52, a1, v7, v8, v9, v10);
  sub_2DACDE0((__int64)v53, (__int64)v52);
  sub_C8CD80((__int64)v65, (__int64)v66, (__int64)v57, v11, v12, v13);
  v17 = v59;
  v18 = (__m128i *)v69;
  v67 = (__m128i *)v69;
  v68 = 0x800000000LL;
  if ( v59 )
  {
    v41 = v59;
    if ( v59 > 8 )
    {
      sub_2DACD40((__int64)&v67, v59, (__int64)v69, (__int64)v14, v15, v16);
      v18 = v67;
      v41 = v59;
    }
    v14 = v58;
    v42 = (const __m128i *)((char *)v58 + 24 * v41);
    if ( v58 != v42 )
    {
      do
      {
        if ( v18 )
        {
          *v18 = _mm_loadu_si128(v14);
          v18[1].m128i_i64[0] = v14[1].m128i_i64[0];
        }
        v14 = (const __m128i *)((char *)v14 + 24);
        v18 = (__m128i *)((char *)v18 + 24);
      }
      while ( v42 != v14 );
    }
    LODWORD(v68) = v17;
  }
  sub_C8CD80((__int64)v60, (__int64)v61, (__int64)v53, (__int64)v14, v15, v16);
  v22 = v55;
  v23 = (__m128i *)v64;
  v62 = (__m128i *)v64;
  v63 = 0x800000000LL;
  if ( v55 )
  {
    v47 = v55;
    if ( v55 > 8 )
    {
      sub_2DACD40((__int64)&v62, v55, (__int64)v64, (__int64)v19, v20, v21);
      v23 = v62;
      v47 = v55;
    }
    v19 = v54;
    v48 = (const __m128i *)((char *)v54 + 24 * v47);
    if ( v54 != v48 )
    {
      do
      {
        if ( v23 )
        {
          *v23 = _mm_loadu_si128(v19);
          v23[1].m128i_i64[0] = v19[1].m128i_i64[0];
        }
        v19 = (const __m128i *)((char *)v19 + 24);
        v23 = (__m128i *)((char *)v23 + 24);
      }
      while ( v48 != v19 );
    }
    LODWORD(v63) = v22;
  }
  sub_C8CD80((__int64)v75, (__int64)v76, (__int64)v65, (__int64)v19, v20, v21);
  v27 = v68;
  v28 = (__m128i *)v79;
  v77 = (__m128i *)v79;
  v78 = 0x800000000LL;
  if ( (_DWORD)v68 )
  {
    v43 = (unsigned int)v68;
    if ( (unsigned int)v68 > 8 )
    {
      sub_2DACD40((__int64)&v77, (unsigned int)v68, (__int64)v79, (__int64)v24, v25, v26);
      v28 = v77;
      v43 = (unsigned int)v68;
    }
    v24 = v67;
    v44 = (__m128i *)((char *)v67 + 24 * v43);
    if ( v67 != v44 )
    {
      do
      {
        if ( v28 )
        {
          *v28 = _mm_loadu_si128(v24);
          v28[1].m128i_i64[0] = v24[1].m128i_i64[0];
        }
        v24 = (const __m128i *)((char *)v24 + 24);
        v28 = (__m128i *)((char *)v28 + 24);
      }
      while ( v44 != v24 );
    }
    LODWORD(v78) = v27;
  }
  sub_C8CD80((__int64)v70, (__int64)v71, (__int64)v60, (__int64)v24, v25, v26);
  v31 = v63;
  v32 = (__m128i *)v74;
  v72 = (__m128i *)v74;
  v73 = 0x800000000LL;
  if ( (_DWORD)v63 )
  {
    v33 = (unsigned int)v63;
    v45 = (unsigned int)v63;
    if ( (unsigned int)v63 > 8 )
    {
      v50 = v63;
      v51 = (unsigned int)v63;
      sub_2DACD40((__int64)&v72, (unsigned int)v63, v29, (__int64)v74, (unsigned int)v63, v30);
      v32 = v72;
      v45 = (unsigned int)v63;
      v31 = v50;
      v33 = v51;
    }
    v46 = v62;
    v30 = (__int64)&v62->m128i_i64[3 * v45];
    if ( v62 != (__m128i *)v30 )
    {
      do
      {
        if ( v32 )
        {
          *v32 = _mm_loadu_si128(v46);
          v32[1].m128i_i64[0] = v46[1].m128i_i64[0];
        }
        v46 = (const __m128i *)((char *)v46 + 24);
        v32 = (__m128i *)((char *)v32 + 24);
      }
      while ( (const __m128i *)v30 != v46 );
      v32 = v72;
    }
    LODWORD(v73) = v31;
  }
  else
  {
    v33 = 0;
  }
  while ( 1 )
  {
    v34 = 24 * v33;
    if ( v33 != (unsigned int)v78 )
      goto LABEL_10;
    v33 = (__int64)v77;
    if ( v32 == (__m128i *)&v32->m128i_i8[v34] )
      break;
    v35 = (__int64 *)v32;
    while ( v35[2] == *(_QWORD *)(v33 + 16) )
    {
      if ( v35[1] != *(_QWORD *)(v33 + 8) )
        break;
      v30 = *(_QWORD *)v33;
      if ( *v35 != *(_QWORD *)v33 )
        break;
      v35 += 3;
      v33 += 24;
      if ( &v32->m128i_i8[v34] == (__int8 *)v35 )
        goto LABEL_18;
    }
LABEL_10:
    v36 = v32->m128i_i64[(unsigned __int64)v34 / 8 - 1];
    v37 = *(unsigned int *)(a3 + 8);
    v38 = *(unsigned int *)(a3 + 12);
    if ( v37 + 1 > v38 )
    {
      v33 = a3 + 16;
      v49 = v36;
      sub_C8D5F0(a3, (const void *)(a3 + 16), v37 + 1, 8u, v36, v30);
      v37 = *(unsigned int *)(a3 + 8);
      v36 = v49;
    }
    v39 = *(_QWORD *)a3;
    *(_QWORD *)(*(_QWORD *)a3 + 8 * v37) = v36;
    ++*(_DWORD *)(a3 + 8);
    LODWORD(v73) = v73 - 1;
    if ( (_DWORD)v73 )
    {
      sub_2DACB60((__int64)v70, v33, v39, v38, v36, v30);
      v32 = v72;
      v33 = (unsigned int)v73;
    }
    else
    {
      v32 = v72;
      v33 = 0;
    }
  }
LABEL_18:
  sub_2E507D0((__int64)v70);
  sub_2E507D0((__int64)v75);
  sub_2E507D0((__int64)v60);
  sub_2E507D0((__int64)v65);
  sub_2E507D0((__int64)v53);
  sub_2E507D0((__int64)v52);
  sub_2E507D0((__int64)v57);
  sub_2E507D0((__int64)v56);
  return a3;
}
