// Function: sub_2E592C0
// Address: 0x2e592c0
//
void __fastcall sub_2E592C0(__int64 *a1)
{
  __int64 v2; // rdx
  __int64 v3; // rsi
  __int64 v4; // rcx
  __int64 v5; // r8
  __int64 v6; // r9
  __int64 v7; // rdx
  __int64 v8; // r8
  __int64 v9; // r9
  __m128i *v10; // rax
  __int64 v11; // rcx
  __int64 v12; // rdx
  __int64 v13; // r8
  __int64 v14; // r9
  __m128i *v15; // rax
  unsigned int v16; // ecx
  __int64 v17; // rsi
  __int64 v18; // rax
  __int64 v19; // rdx
  __int64 *v20; // rsi
  __int64 *v21; // rax
  __int64 v22; // rsi
  __int64 v23; // rdx
  __int64 v24; // rcx
  __int64 v25; // r8
  __int64 v26; // r9
  bool v27; // zf
  __int64 *v28; // rdi
  __int64 *v29; // r13
  __int64 *v30; // r12
  __int64 v31; // rsi
  __int64 v32; // r12
  __int64 i; // r13
  __int64 v34; // rsi
  const __m128i *v35; // rdx
  const __m128i *v36; // rdi
  __int64 v37; // rdi
  const __m128i *v38; // rdx
  const __m128i *v39; // r8
  unsigned int v40; // [rsp+0h] [rbp-760h]
  unsigned int v41; // [rsp+8h] [rbp-758h]
  __int64 v42; // [rsp+8h] [rbp-758h]
  _QWORD v43[38]; // [rsp+10h] [rbp-750h] BYREF
  _BYTE v44[304]; // [rsp+140h] [rbp-620h] BYREF
  _BYTE v45[32]; // [rsp+270h] [rbp-4F0h] BYREF
  char v46[64]; // [rsp+290h] [rbp-4D0h] BYREF
  __m128i *v47; // [rsp+2D0h] [rbp-490h] BYREF
  __int64 v48; // [rsp+2D8h] [rbp-488h]
  _BYTE v49[192]; // [rsp+2E0h] [rbp-480h] BYREF
  _BYTE v50[32]; // [rsp+3A0h] [rbp-3C0h] BYREF
  char v51[64]; // [rsp+3C0h] [rbp-3A0h] BYREF
  __m128i *v52; // [rsp+400h] [rbp-360h] BYREF
  __int64 v53; // [rsp+408h] [rbp-358h]
  _BYTE v54[192]; // [rsp+410h] [rbp-350h] BYREF
  __int64 *v55; // [rsp+4D0h] [rbp-290h] BYREF
  int v56; // [rsp+4D8h] [rbp-288h]
  char v57; // [rsp+4E0h] [rbp-280h] BYREF
  const __m128i *v58; // [rsp+530h] [rbp-230h]
  unsigned int v59; // [rsp+538h] [rbp-228h]
  _BYTE v60[96]; // [rsp+600h] [rbp-160h] BYREF
  const __m128i *v61; // [rsp+660h] [rbp-100h]
  unsigned int v62; // [rsp+668h] [rbp-F8h]

  v2 = *a1;
  memset(v43, 0, sizeof(v43));
  v3 = *(_QWORD *)(v2 + 328);
  v43[1] = &v43[4];
  v43[12] = &v43[14];
  LODWORD(v43[2]) = 8;
  BYTE4(v43[3]) = 1;
  HIDWORD(v43[13]) = 8;
  sub_2E56370((__int64)v44, v3, v2, 0, (__int64)v44, (__int64)v43);
  sub_2DACDE0((__int64)v50, (__int64)v43);
  sub_2DACDE0((__int64)v45, (__int64)v44);
  sub_2DACDE0((__int64)&v55, (__int64)v45);
  sub_2DACDE0((__int64)v60, (__int64)v50);
  sub_2E507D0((__int64)v45);
  sub_2E507D0((__int64)v50);
  sub_2E507D0((__int64)v44);
  sub_2E507D0((__int64)v43);
  sub_C8CD80((__int64)v45, (__int64)v46, (__int64)&v55, v4, v5, v6);
  v10 = (__m128i *)v49;
  v48 = 0x800000000LL;
  v11 = v59;
  v47 = (__m128i *)v49;
  if ( v59 )
  {
    v34 = v59;
    if ( v59 > 8 )
    {
      v41 = v59;
      sub_2DACD40((__int64)&v47, v59, v7, v59, v8, v9);
      v10 = v47;
      v34 = v59;
      v11 = v41;
    }
    v35 = v58;
    v36 = (const __m128i *)((char *)v58 + 24 * v34);
    if ( v58 != v36 )
    {
      do
      {
        if ( v10 )
        {
          *v10 = _mm_loadu_si128(v35);
          v10[1].m128i_i64[0] = v35[1].m128i_i64[0];
        }
        v35 = (const __m128i *)((char *)v35 + 24);
        v10 = (__m128i *)((char *)v10 + 24);
      }
      while ( v36 != v35 );
    }
    LODWORD(v48) = v11;
  }
  sub_C8CD80((__int64)v50, (__int64)v51, (__int64)v60, v11, v8, v9);
  v15 = (__m128i *)v54;
  v53 = 0x800000000LL;
  v16 = v62;
  v52 = (__m128i *)v54;
  if ( v62 )
  {
    v17 = v62;
    v37 = v62;
    if ( v62 > 8 )
    {
      v40 = v62;
      v42 = v62;
      sub_2DACD40((__int64)&v52, v62, v12, v62, v13, v14);
      v15 = v52;
      v37 = v62;
      v16 = v40;
      v17 = v42;
    }
    v38 = v61;
    v39 = (const __m128i *)((char *)v61 + 24 * v37);
    if ( v61 != v39 )
    {
      do
      {
        if ( v15 )
        {
          *v15 = _mm_loadu_si128(v38);
          v15[1].m128i_i64[0] = v38[1].m128i_i64[0];
        }
        v38 = (const __m128i *)((char *)v38 + 24);
        v15 = (__m128i *)((char *)v15 + 24);
      }
      while ( v39 != v38 );
    }
    LODWORD(v53) = v16;
  }
  else
  {
    v17 = 0;
  }
LABEL_4:
  v18 = (unsigned int)v48;
  while ( 1 )
  {
    v19 = 24 * v18;
    if ( v18 != v17 )
      goto LABEL_9;
    v20 = (__int64 *)v52;
    if ( v47 == (__m128i *)&v47->m128i_i8[v19] )
      break;
    v21 = (__int64 *)v47;
    while ( v21[2] == v20[2] && v21[1] == v20[1] && *v21 == *v20 )
    {
      v21 += 3;
      v20 += 3;
      if ( &v47->m128i_i8[v19] == (__int8 *)v21 )
        goto LABEL_15;
    }
LABEL_9:
    v22 = v47->m128i_i64[(unsigned __int64)v19 / 8 - 1];
    sub_2E58D20((__int64)a1, v22);
    v27 = (_DWORD)v48 == 1;
    v18 = (unsigned int)(v48 - 1);
    LODWORD(v48) = v48 - 1;
    if ( !v27 )
    {
      sub_2DACB60((__int64)v45, v22, v23, v24, v25, v26);
      v17 = (unsigned int)v53;
      goto LABEL_4;
    }
    v17 = (unsigned int)v53;
  }
LABEL_15:
  sub_2E507D0((__int64)v50);
  sub_2E507D0((__int64)v45);
  sub_2E507D0((__int64)v60);
  sub_2E507D0((__int64)&v55);
  sub_2EA5BA0(&v55, a1[1]);
  v28 = v55;
  v29 = &v55[v56];
  v30 = v55;
  if ( v29 != v55 )
  {
    do
    {
      v31 = *v30++;
      sub_2E57DC0((__int64)a1, v31);
    }
    while ( v29 != v30 );
    v28 = v55;
  }
  if ( v28 != (__int64 *)&v57 )
    _libc_free((unsigned __int64)v28);
  v32 = *(_QWORD *)(*a1 + 328);
  for ( i = *a1 + 320; i != v32; v32 = *(_QWORD *)(v32 + 8) )
    sub_2E58AA0((__int64)a1, v32);
}
