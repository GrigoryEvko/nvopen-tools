// Function: sub_3083F80
// Address: 0x3083f80
//
void __fastcall sub_3083F80(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  __int64 v6; // rsi
  __int64 v7; // rcx
  __int64 v8; // r8
  __int64 v9; // r9
  __int64 v10; // rcx
  __int64 v11; // r8
  __int64 v12; // r9
  int v13; // r12d
  __int64 v14; // rcx
  __int64 v15; // r8
  __int64 v16; // rcx
  __int64 v17; // r8
  unsigned int v18; // r15d
  __int64 v19; // rcx
  __int64 v20; // r8
  __int64 v21; // r9
  __int64 v22; // rsi
  __m128i *v23; // rdx
  const __m128i *v24; // rax
  unsigned __int64 v25; // rdi
  __int64 v26; // rsi
  __m128i *v27; // rdx
  const __m128i *v28; // rax
  const __m128i *v29; // rsi
  _BYTE v31[8]; // [rsp+10h] [rbp-750h] BYREF
  unsigned __int64 v32; // [rsp+18h] [rbp-748h]
  char v33; // [rsp+2Ch] [rbp-734h]
  const __m128i *v34; // [rsp+70h] [rbp-6F0h]
  unsigned int v35; // [rsp+78h] [rbp-6E8h]
  char v36; // [rsp+80h] [rbp-6E0h] BYREF
  unsigned __int64 v37[38]; // [rsp+140h] [rbp-620h] BYREF
  _BYTE v38[8]; // [rsp+270h] [rbp-4F0h] BYREF
  unsigned __int64 v39; // [rsp+278h] [rbp-4E8h]
  char v40; // [rsp+28Ch] [rbp-4D4h]
  _BYTE v41[64]; // [rsp+290h] [rbp-4D0h] BYREF
  __m128i *v42; // [rsp+2D0h] [rbp-490h] BYREF
  __int64 v43; // [rsp+2D8h] [rbp-488h]
  _BYTE v44[192]; // [rsp+2E0h] [rbp-480h] BYREF
  _BYTE v45[8]; // [rsp+3A0h] [rbp-3C0h] BYREF
  unsigned __int64 v46; // [rsp+3A8h] [rbp-3B8h]
  char v47; // [rsp+3BCh] [rbp-3A4h]
  char *v48; // [rsp+400h] [rbp-360h]
  char v49; // [rsp+410h] [rbp-350h] BYREF
  _BYTE v50[8]; // [rsp+4D0h] [rbp-290h] BYREF
  unsigned __int64 v51; // [rsp+4D8h] [rbp-288h]
  char v52; // [rsp+4ECh] [rbp-274h]
  _BYTE v53[64]; // [rsp+4F0h] [rbp-270h] BYREF
  __m128i *v54; // [rsp+530h] [rbp-230h] BYREF
  __int64 v55; // [rsp+538h] [rbp-228h]
  _BYTE v56[192]; // [rsp+540h] [rbp-220h] BYREF
  _BYTE v57[8]; // [rsp+600h] [rbp-160h] BYREF
  unsigned __int64 v58; // [rsp+608h] [rbp-158h]
  char v59; // [rsp+61Ch] [rbp-144h]
  char *v60; // [rsp+660h] [rbp-100h]
  char v61; // [rsp+670h] [rbp-F0h] BYREF

  v6 = *(_QWORD *)(a2 + 328);
  memset(v37, 0, sizeof(v37));
  LODWORD(v37[2]) = 8;
  v37[1] = (unsigned __int64)&v37[4];
  BYTE4(v37[3]) = 1;
  v37[12] = (unsigned __int64)&v37[14];
  HIDWORD(v37[13]) = 8;
  sub_2E56370((__int64)v31, v6, a3, 0, a5, a6);
  sub_C8CD80((__int64)v50, (__int64)v53, (__int64)v37, v7, v8, v9);
  v13 = v37[13];
  v54 = (__m128i *)v56;
  v55 = 0x800000000LL;
  if ( LODWORD(v37[13]) )
  {
    v22 = LODWORD(v37[13]);
    v23 = (__m128i *)v56;
    if ( LODWORD(v37[13]) > 8 )
    {
      sub_2DACD40((__int64)&v54, LODWORD(v37[13]), (__int64)v56, v10, v11, v12);
      v23 = v54;
      v22 = LODWORD(v37[13]);
    }
    v24 = (const __m128i *)v37[12];
    v25 = v37[12] + 24 * v22;
    if ( v37[12] != v25 )
    {
      do
      {
        if ( v23 )
        {
          *v23 = _mm_loadu_si128(v24);
          v23[1].m128i_i64[0] = v24[1].m128i_i64[0];
        }
        v24 = (const __m128i *)((char *)v24 + 24);
        v23 = (__m128i *)((char *)v23 + 24);
      }
      while ( (const __m128i *)v25 != v24 );
    }
    LODWORD(v55) = v13;
  }
  sub_2DACDE0((__int64)v57, (__int64)v50);
  sub_C8CD80((__int64)v38, (__int64)v41, (__int64)v31, v14, v15, (__int64)v38);
  v18 = v35;
  v42 = (__m128i *)v44;
  v43 = 0x800000000LL;
  if ( v35 )
  {
    v26 = v35;
    v27 = (__m128i *)v44;
    if ( v35 > 8 )
    {
      sub_2DACD40((__int64)&v42, v35, (__int64)v44, v16, v17, (__int64)v38);
      v27 = v42;
      v26 = v35;
    }
    v28 = v34;
    v29 = (const __m128i *)((char *)v34 + 24 * v26);
    if ( v34 != v29 )
    {
      do
      {
        if ( v27 )
        {
          *v27 = _mm_loadu_si128(v28);
          v27[1].m128i_i64[0] = v28[1].m128i_i64[0];
        }
        v28 = (const __m128i *)((char *)v28 + 24);
        v27 = (__m128i *)((char *)v27 + 24);
      }
      while ( v29 != v28 );
    }
    LODWORD(v43) = v18;
  }
  sub_2DACDE0((__int64)v45, (__int64)v38);
  sub_2E564A0((__int64)v45, (__int64)v57, a1, v19, v20, v21);
  if ( v48 != &v49 )
    _libc_free((unsigned __int64)v48);
  if ( !v47 )
    _libc_free(v46);
  if ( v42 != (__m128i *)v44 )
    _libc_free((unsigned __int64)v42);
  if ( !v40 )
    _libc_free(v39);
  if ( v60 != &v61 )
    _libc_free((unsigned __int64)v60);
  if ( !v59 )
    _libc_free(v58);
  if ( v54 != (__m128i *)v56 )
    _libc_free((unsigned __int64)v54);
  if ( !v52 )
    _libc_free(v51);
  if ( v34 != (const __m128i *)&v36 )
    _libc_free((unsigned __int64)v34);
  if ( !v33 )
    _libc_free(v32);
  if ( (unsigned __int64 *)v37[12] != &v37[14] )
    _libc_free(v37[12]);
  if ( !BYTE4(v37[3]) )
    _libc_free(v37[1]);
}
