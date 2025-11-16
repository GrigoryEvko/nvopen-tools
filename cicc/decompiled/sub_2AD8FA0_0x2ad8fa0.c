// Function: sub_2AD8FA0
// Address: 0x2ad8fa0
//
__int64 __fastcall sub_2AD8FA0(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
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
  __int64 *v35; // rsi
  __int64 *v36; // rdx
  __int64 v37; // r8
  __int64 v38; // rax
  __int64 v40; // rsi
  const __m128i *v41; // rdi
  __int64 v42; // rsi
  const __m128i *v43; // rdi
  __int64 v44; // rdi
  const __m128i *v45; // rdx
  __int64 v46; // rsi
  const __m128i *v47; // rdi
  __int64 v48; // [rsp+8h] [rbp-9D8h]
  int v49; // [rsp+8h] [rbp-9D8h]
  __int64 v50; // [rsp+10h] [rbp-9D0h]
  _BYTE v51[304]; // [rsp+30h] [rbp-9B0h] BYREF
  _BYTE v52[96]; // [rsp+160h] [rbp-880h] BYREF
  const __m128i *v53; // [rsp+1C0h] [rbp-820h]
  unsigned int v54; // [rsp+1C8h] [rbp-818h]
  _BYTE v55[304]; // [rsp+290h] [rbp-750h] BYREF
  _BYTE v56[96]; // [rsp+3C0h] [rbp-620h] BYREF
  const __m128i *v57; // [rsp+420h] [rbp-5C0h]
  unsigned int v58; // [rsp+428h] [rbp-5B8h]
  _BYTE v59[32]; // [rsp+4F0h] [rbp-4F0h] BYREF
  _BYTE v60[64]; // [rsp+510h] [rbp-4D0h] BYREF
  __m128i *v61; // [rsp+550h] [rbp-490h] BYREF
  __int64 v62; // [rsp+558h] [rbp-488h]
  _BYTE v63[192]; // [rsp+560h] [rbp-480h] BYREF
  _BYTE v64[32]; // [rsp+620h] [rbp-3C0h] BYREF
  _BYTE v65[64]; // [rsp+640h] [rbp-3A0h] BYREF
  __m128i *v66; // [rsp+680h] [rbp-360h] BYREF
  __int64 v67; // [rsp+688h] [rbp-358h]
  _BYTE v68[192]; // [rsp+690h] [rbp-350h] BYREF
  _BYTE v69[32]; // [rsp+750h] [rbp-290h] BYREF
  _BYTE v70[64]; // [rsp+770h] [rbp-270h] BYREF
  __m128i *v71; // [rsp+7B0h] [rbp-230h] BYREF
  __int64 v72; // [rsp+7B8h] [rbp-228h]
  _BYTE v73[192]; // [rsp+7C0h] [rbp-220h] BYREF
  _BYTE v74[32]; // [rsp+880h] [rbp-160h] BYREF
  _BYTE v75[64]; // [rsp+8A0h] [rbp-140h] BYREF
  __m128i *v76; // [rsp+8E0h] [rbp-100h] BYREF
  __int64 v77; // [rsp+8E8h] [rbp-F8h]
  _BYTE v78[240]; // [rsp+8F0h] [rbp-F0h] BYREF

  sub_2AD8EE0((__int64)v55, a2, a3, a4, a5, a6);
  sub_2AD8DC0((__int64)v56, (__int64)v55);
  sub_2AD8EE0((__int64)v51, a1, v7, v8, v9, v10);
  sub_2AD8DC0((__int64)v52, (__int64)v51);
  sub_C8CD80((__int64)v64, (__int64)v65, (__int64)v56, v11, v12, v13);
  v17 = v58;
  v18 = (__m128i *)v68;
  v66 = (__m128i *)v68;
  v67 = 0x800000000LL;
  if ( v58 )
  {
    v40 = v58;
    if ( v58 > 8 )
    {
      sub_2AD8D20((__int64)&v66, v58, (__int64)v68, (__int64)v14, v15, v16);
      v18 = v66;
      v40 = v58;
    }
    v14 = v57;
    v41 = (const __m128i *)((char *)v57 + 24 * v40);
    if ( v57 != v41 )
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
      while ( v41 != v14 );
    }
    LODWORD(v67) = v17;
  }
  sub_C8CD80((__int64)v59, (__int64)v60, (__int64)v52, (__int64)v14, v15, v16);
  v22 = v54;
  v23 = (__m128i *)v63;
  v61 = (__m128i *)v63;
  v62 = 0x800000000LL;
  if ( v54 )
  {
    v46 = v54;
    if ( v54 > 8 )
    {
      sub_2AD8D20((__int64)&v61, v54, (__int64)v63, (__int64)v19, v20, v21);
      v23 = v61;
      v46 = v54;
    }
    v19 = v53;
    v47 = (const __m128i *)((char *)v53 + 24 * v46);
    if ( v53 != v47 )
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
      while ( v47 != v19 );
    }
    LODWORD(v62) = v22;
  }
  sub_C8CD80((__int64)v74, (__int64)v75, (__int64)v64, (__int64)v19, v20, v21);
  v27 = v67;
  v28 = (__m128i *)v78;
  v76 = (__m128i *)v78;
  v77 = 0x800000000LL;
  if ( (_DWORD)v67 )
  {
    v42 = (unsigned int)v67;
    if ( (unsigned int)v67 > 8 )
    {
      sub_2AD8D20((__int64)&v76, (unsigned int)v67, (__int64)v78, (__int64)v24, v25, v26);
      v28 = v76;
      v42 = (unsigned int)v67;
    }
    v24 = v66;
    v43 = (__m128i *)((char *)v66 + 24 * v42);
    if ( v66 != v43 )
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
      while ( v43 != v24 );
    }
    LODWORD(v77) = v27;
  }
  sub_C8CD80((__int64)v69, (__int64)v70, (__int64)v59, (__int64)v24, v25, v26);
  v31 = v62;
  v32 = (__m128i *)v73;
  v71 = (__m128i *)v73;
  v72 = 0x800000000LL;
  if ( (_DWORD)v62 )
  {
    v33 = (unsigned int)v62;
    v44 = (unsigned int)v62;
    if ( (unsigned int)v62 > 8 )
    {
      v49 = v62;
      v50 = (unsigned int)v62;
      sub_2AD8D20((__int64)&v71, (unsigned int)v62, v29, (__int64)v73, (unsigned int)v62, v30);
      v32 = v71;
      v44 = (unsigned int)v62;
      v31 = v49;
      v33 = v50;
    }
    v45 = v61;
    v30 = (__int64)&v61->m128i_i64[3 * v44];
    if ( v61 != (__m128i *)v30 )
    {
      do
      {
        if ( v32 )
        {
          *v32 = _mm_loadu_si128(v45);
          v32[1].m128i_i64[0] = v45[1].m128i_i64[0];
        }
        v45 = (const __m128i *)((char *)v45 + 24);
        v32 = (__m128i *)((char *)v32 + 24);
      }
      while ( (const __m128i *)v30 != v45 );
      v32 = v71;
    }
    LODWORD(v72) = v31;
  }
  else
  {
    v33 = 0;
  }
  while ( 1 )
  {
    v34 = 24 * v33;
    if ( v33 != (unsigned int)v77 )
      goto LABEL_10;
    v35 = (__int64 *)v76;
    if ( v32 == (__m128i *)&v32->m128i_i8[v34] )
      break;
    v36 = (__int64 *)v32;
    while ( v36[2] == v35[2] )
    {
      if ( v36[1] != v35[1] )
        break;
      v30 = *v35;
      if ( *v36 != *v35 )
        break;
      v36 += 3;
      v35 += 3;
      if ( &v32->m128i_i8[v34] == (__int8 *)v36 )
        goto LABEL_18;
    }
LABEL_10:
    v37 = v32->m128i_i64[(unsigned __int64)v34 / 8 - 1];
    v38 = *(unsigned int *)(a3 + 8);
    if ( v38 + 1 > (unsigned __int64)*(unsigned int *)(a3 + 12) )
    {
      v48 = v37;
      sub_C8D5F0(a3, (const void *)(a3 + 16), v38 + 1, 8u, v37, v30);
      v38 = *(unsigned int *)(a3 + 8);
      v37 = v48;
    }
    *(_QWORD *)(*(_QWORD *)a3 + 8 * v38) = v37;
    ++*(_DWORD *)(a3 + 8);
    LODWORD(v72) = v72 - 1;
    if ( (_DWORD)v72 )
    {
      sub_2AD8BC0((__int64)v69);
      v32 = v71;
      v33 = (unsigned int)v72;
    }
    else
    {
      v32 = v71;
      v33 = 0;
    }
  }
LABEL_18:
  sub_2AC2230((__int64)v69);
  sub_2AC2230((__int64)v74);
  sub_2AC2230((__int64)v59);
  sub_2AC2230((__int64)v64);
  sub_2AC2230((__int64)v52);
  sub_2AC2230((__int64)v51);
  sub_2AC2230((__int64)v56);
  sub_2AC2230((__int64)v55);
  return a3;
}
