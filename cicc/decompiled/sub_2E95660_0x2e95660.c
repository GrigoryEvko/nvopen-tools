// Function: sub_2E95660
// Address: 0x2e95660
//
void __fastcall sub_2E95660(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  __int64 v6; // rax
  __int64 v7; // rdx
  __int64 v8; // rcx
  __int64 v9; // rcx
  __int64 v10; // r8
  __int64 v11; // rcx
  __int64 v12; // r8
  int v13; // r12d
  __int64 v14; // rcx
  __int64 v15; // r8
  __int64 v16; // r8
  unsigned int v17; // ecx
  __int64 v18; // rcx
  __int64 v19; // r8
  __int64 v20; // r9
  __int64 v21; // rsi
  __m128i *v22; // rdx
  const __m128i *v23; // rax
  unsigned __int64 v24; // rdi
  __int64 v25; // rsi
  __m128i *v26; // rdx
  const __m128i *v27; // rax
  const __m128i *v28; // rdi
  unsigned int v29; // [rsp+10h] [rbp-760h]
  __int64 v31; // [rsp+20h] [rbp-750h] BYREF
  __int64 *v32; // [rsp+28h] [rbp-748h]
  int v33; // [rsp+30h] [rbp-740h]
  int v34; // [rsp+34h] [rbp-73Ch]
  int v35; // [rsp+38h] [rbp-738h]
  char v36; // [rsp+3Ch] [rbp-734h]
  __int64 v37; // [rsp+40h] [rbp-730h] BYREF
  __int64 *v38; // [rsp+80h] [rbp-6F0h]
  unsigned int v39; // [rsp+88h] [rbp-6E8h]
  int v40; // [rsp+8Ch] [rbp-6E4h]
  __int64 v41[24]; // [rsp+90h] [rbp-6E0h] BYREF
  unsigned __int64 v42[38]; // [rsp+150h] [rbp-620h] BYREF
  char v43[8]; // [rsp+280h] [rbp-4F0h] BYREF
  unsigned __int64 v44; // [rsp+288h] [rbp-4E8h]
  char v45; // [rsp+29Ch] [rbp-4D4h]
  char v46[64]; // [rsp+2A0h] [rbp-4D0h] BYREF
  __m128i *v47; // [rsp+2E0h] [rbp-490h] BYREF
  __int64 v48; // [rsp+2E8h] [rbp-488h]
  _BYTE v49[192]; // [rsp+2F0h] [rbp-480h] BYREF
  char v50[8]; // [rsp+3B0h] [rbp-3C0h] BYREF
  unsigned __int64 v51; // [rsp+3B8h] [rbp-3B8h]
  char v52; // [rsp+3CCh] [rbp-3A4h]
  char *v53; // [rsp+410h] [rbp-360h]
  char v54; // [rsp+420h] [rbp-350h] BYREF
  char v55[8]; // [rsp+4E0h] [rbp-290h] BYREF
  unsigned __int64 v56; // [rsp+4E8h] [rbp-288h]
  char v57; // [rsp+4FCh] [rbp-274h]
  char v58[64]; // [rsp+500h] [rbp-270h] BYREF
  __m128i *v59; // [rsp+540h] [rbp-230h] BYREF
  __int64 v60; // [rsp+548h] [rbp-228h]
  _BYTE v61[192]; // [rsp+550h] [rbp-220h] BYREF
  char v62[8]; // [rsp+610h] [rbp-160h] BYREF
  unsigned __int64 v63; // [rsp+618h] [rbp-158h]
  char v64; // [rsp+62Ch] [rbp-144h]
  char *v65; // [rsp+670h] [rbp-100h]
  char v66; // [rsp+680h] [rbp-F0h] BYREF

  memset(v42, 0, sizeof(v42));
  v32 = &v37;
  v42[1] = (unsigned __int64)&v42[4];
  v6 = *(_QWORD *)(a2 + 328);
  LODWORD(v42[2]) = 8;
  v7 = *(_QWORD *)(v6 + 112);
  v8 = *(unsigned int *)(v6 + 120);
  v37 = v6;
  v41[2] = v6;
  v41[1] = v7;
  v41[0] = v7 + 8 * v8;
  BYTE4(v42[3]) = 1;
  v42[12] = (unsigned __int64)&v42[14];
  HIDWORD(v42[13]) = 8;
  v33 = 8;
  v35 = 0;
  v36 = 1;
  v38 = v41;
  v40 = 8;
  v34 = 1;
  v31 = 1;
  v39 = 1;
  sub_2DACB60((__int64)&v31, a2, v7, v41[0], a5, a6);
  sub_C8CD80((__int64)v55, (__int64)v58, (__int64)v42, v9, v10, (__int64)v55);
  v13 = v42[13];
  v59 = (__m128i *)v61;
  v60 = 0x800000000LL;
  if ( LODWORD(v42[13]) )
  {
    v21 = LODWORD(v42[13]);
    v22 = (__m128i *)v61;
    if ( LODWORD(v42[13]) > 8 )
    {
      sub_2DACD40((__int64)&v59, LODWORD(v42[13]), (__int64)v61, v11, v12, (__int64)v55);
      v22 = v59;
      v21 = LODWORD(v42[13]);
    }
    v23 = (const __m128i *)v42[12];
    v24 = v42[12] + 24 * v21;
    if ( v42[12] != v24 )
    {
      do
      {
        if ( v22 )
        {
          *v22 = _mm_loadu_si128(v23);
          v22[1].m128i_i64[0] = v23[1].m128i_i64[0];
        }
        v23 = (const __m128i *)((char *)v23 + 24);
        v22 = (__m128i *)((char *)v22 + 24);
      }
      while ( (const __m128i *)v24 != v23 );
    }
    LODWORD(v60) = v13;
  }
  sub_2DACDE0((__int64)v62, (__int64)v55);
  sub_C8CD80((__int64)v43, (__int64)v46, (__int64)&v31, v14, v15, (__int64)v43);
  v17 = v39;
  v47 = (__m128i *)v49;
  v48 = 0x800000000LL;
  if ( v39 )
  {
    v25 = v39;
    v26 = (__m128i *)v49;
    if ( v39 > 8 )
    {
      v29 = v39;
      sub_2DACD40((__int64)&v47, v39, (__int64)v49, v39, v16, (__int64)v43);
      v26 = v47;
      v25 = v39;
      v17 = v29;
    }
    v27 = (const __m128i *)v38;
    v28 = (const __m128i *)&v38[3 * v25];
    if ( v38 != (__int64 *)v28 )
    {
      do
      {
        if ( v26 )
        {
          *v26 = _mm_loadu_si128(v27);
          v26[1].m128i_i64[0] = v27[1].m128i_i64[0];
        }
        v27 = (const __m128i *)((char *)v27 + 24);
        v26 = (__m128i *)((char *)v26 + 24);
      }
      while ( v28 != v27 );
    }
    LODWORD(v48) = v17;
  }
  sub_2DACDE0((__int64)v50, (__int64)v43);
  sub_2E564A0((__int64)v50, (__int64)v62, a1, v18, v19, v20);
  if ( v53 != &v54 )
    _libc_free((unsigned __int64)v53);
  if ( !v52 )
    _libc_free(v51);
  if ( v47 != (__m128i *)v49 )
    _libc_free((unsigned __int64)v47);
  if ( !v45 )
    _libc_free(v44);
  if ( v65 != &v66 )
    _libc_free((unsigned __int64)v65);
  if ( !v64 )
    _libc_free(v63);
  if ( v59 != (__m128i *)v61 )
    _libc_free((unsigned __int64)v59);
  if ( !v57 )
    _libc_free(v56);
  if ( v38 != v41 )
    _libc_free((unsigned __int64)v38);
  if ( !v36 )
    _libc_free((unsigned __int64)v32);
  if ( (unsigned __int64 *)v42[12] != &v42[14] )
    _libc_free(v42[12]);
  if ( !BYTE4(v42[3]) )
    _libc_free(v42[1]);
}
