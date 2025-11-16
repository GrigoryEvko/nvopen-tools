// Function: sub_357E7D0
// Address: 0x357e7d0
//
void __fastcall sub_357E7D0(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  __int64 v6; // rdx
  __int64 v7; // rax
  __int64 v8; // rcx
  __int64 v9; // r8
  __int64 v10; // rcx
  __int64 v11; // r8
  int v12; // r12d
  __int64 v13; // rcx
  __int64 v14; // r8
  __int64 v15; // r8
  unsigned int v16; // ecx
  __int64 v17; // rcx
  __int64 v18; // r8
  __int64 v19; // r9
  __int64 v20; // rsi
  __m128i *v21; // rdx
  const __m128i *v22; // rax
  unsigned __int64 v23; // rdi
  __int64 v24; // rsi
  __m128i *v25; // rdx
  const __m128i *v26; // rax
  const __m128i *v27; // rdi
  unsigned int v28; // [rsp+10h] [rbp-760h]
  __int64 v30; // [rsp+20h] [rbp-750h] BYREF
  __int64 *v31; // [rsp+28h] [rbp-748h]
  int v32; // [rsp+30h] [rbp-740h]
  int v33; // [rsp+34h] [rbp-73Ch]
  int v34; // [rsp+38h] [rbp-738h]
  char v35; // [rsp+3Ch] [rbp-734h]
  __int64 v36; // [rsp+40h] [rbp-730h] BYREF
  __int64 *v37; // [rsp+80h] [rbp-6F0h]
  unsigned int v38; // [rsp+88h] [rbp-6E8h]
  int v39; // [rsp+8Ch] [rbp-6E4h]
  __int64 v40[24]; // [rsp+90h] [rbp-6E0h] BYREF
  unsigned __int64 v41[38]; // [rsp+150h] [rbp-620h] BYREF
  char v42[8]; // [rsp+280h] [rbp-4F0h] BYREF
  unsigned __int64 v43; // [rsp+288h] [rbp-4E8h]
  char v44; // [rsp+29Ch] [rbp-4D4h]
  char v45[64]; // [rsp+2A0h] [rbp-4D0h] BYREF
  __m128i *v46; // [rsp+2E0h] [rbp-490h] BYREF
  __int64 v47; // [rsp+2E8h] [rbp-488h]
  _BYTE v48[192]; // [rsp+2F0h] [rbp-480h] BYREF
  char v49[8]; // [rsp+3B0h] [rbp-3C0h] BYREF
  unsigned __int64 v50; // [rsp+3B8h] [rbp-3B8h]
  char v51; // [rsp+3CCh] [rbp-3A4h]
  char *v52; // [rsp+410h] [rbp-360h]
  char v53; // [rsp+420h] [rbp-350h] BYREF
  char v54[8]; // [rsp+4E0h] [rbp-290h] BYREF
  unsigned __int64 v55; // [rsp+4E8h] [rbp-288h]
  char v56; // [rsp+4FCh] [rbp-274h]
  char v57[64]; // [rsp+500h] [rbp-270h] BYREF
  __m128i *v58; // [rsp+540h] [rbp-230h] BYREF
  __int64 v59; // [rsp+548h] [rbp-228h]
  _BYTE v60[192]; // [rsp+550h] [rbp-220h] BYREF
  char v61[8]; // [rsp+610h] [rbp-160h] BYREF
  unsigned __int64 v62; // [rsp+618h] [rbp-158h]
  char v63; // [rsp+62Ch] [rbp-144h]
  char *v64; // [rsp+670h] [rbp-100h]
  char v65; // [rsp+680h] [rbp-F0h] BYREF

  v6 = *(unsigned int *)(a2 + 120);
  memset(v41, 0, sizeof(v41));
  v36 = a2;
  v41[1] = (unsigned __int64)&v41[4];
  v31 = &v36;
  v7 = *(_QWORD *)(a2 + 112);
  v40[2] = a2;
  v40[1] = v7;
  v40[0] = v7 + 8 * v6;
  LODWORD(v41[2]) = 8;
  BYTE4(v41[3]) = 1;
  v41[12] = (unsigned __int64)&v41[14];
  HIDWORD(v41[13]) = 8;
  v32 = 8;
  v34 = 0;
  v35 = 1;
  v37 = v40;
  v39 = 8;
  v33 = 1;
  v30 = 1;
  v38 = 1;
  sub_2EA7130((__int64)&v30, a2, v40[0], 0, a5, a6);
  sub_C8CD80((__int64)v54, (__int64)v57, (__int64)v41, v8, v9, (__int64)v54);
  v12 = v41[13];
  v58 = (__m128i *)v60;
  v59 = 0x800000000LL;
  if ( LODWORD(v41[13]) )
  {
    v20 = LODWORD(v41[13]);
    v21 = (__m128i *)v60;
    if ( LODWORD(v41[13]) > 8 )
    {
      sub_2DACD40((__int64)&v58, LODWORD(v41[13]), (__int64)v60, v10, v11, (__int64)v54);
      v21 = v58;
      v20 = LODWORD(v41[13]);
    }
    v22 = (const __m128i *)v41[12];
    v23 = v41[12] + 24 * v20;
    if ( v41[12] != v23 )
    {
      do
      {
        if ( v21 )
        {
          *v21 = _mm_loadu_si128(v22);
          v21[1].m128i_i64[0] = v22[1].m128i_i64[0];
        }
        v22 = (const __m128i *)((char *)v22 + 24);
        v21 = (__m128i *)((char *)v21 + 24);
      }
      while ( (const __m128i *)v23 != v22 );
    }
    LODWORD(v59) = v12;
  }
  sub_2EA7B20((__int64)v61, (__int64)v54);
  sub_C8CD80((__int64)v42, (__int64)v45, (__int64)&v30, v13, v14, (__int64)v42);
  v16 = v38;
  v46 = (__m128i *)v48;
  v47 = 0x800000000LL;
  if ( v38 )
  {
    v24 = v38;
    v25 = (__m128i *)v48;
    if ( v38 > 8 )
    {
      v28 = v38;
      sub_2DACD40((__int64)&v46, v38, (__int64)v48, v38, v15, (__int64)v42);
      v25 = v46;
      v24 = v38;
      v16 = v28;
    }
    v26 = (const __m128i *)v37;
    v27 = (const __m128i *)&v37[3 * v24];
    if ( v37 != (__int64 *)v27 )
    {
      do
      {
        if ( v25 )
        {
          *v25 = _mm_loadu_si128(v26);
          v25[1].m128i_i64[0] = v26[1].m128i_i64[0];
        }
        v26 = (const __m128i *)((char *)v26 + 24);
        v25 = (__m128i *)((char *)v25 + 24);
      }
      while ( v27 != v26 );
    }
    LODWORD(v47) = v16;
  }
  sub_2EA7B20((__int64)v49, (__int64)v42);
  sub_357E170((__int64)v49, (__int64)v61, a1, v17, v18, v19);
  if ( v52 != &v53 )
    _libc_free((unsigned __int64)v52);
  if ( !v51 )
    _libc_free(v50);
  if ( v46 != (__m128i *)v48 )
    _libc_free((unsigned __int64)v46);
  if ( !v44 )
    _libc_free(v43);
  if ( v64 != &v65 )
    _libc_free((unsigned __int64)v64);
  if ( !v63 )
    _libc_free(v62);
  if ( v58 != (__m128i *)v60 )
    _libc_free((unsigned __int64)v58);
  if ( !v56 )
    _libc_free(v55);
  if ( v37 != v40 )
    _libc_free((unsigned __int64)v37);
  if ( !v35 )
    _libc_free((unsigned __int64)v31);
  if ( (unsigned __int64 *)v41[12] != &v41[14] )
    _libc_free(v41[12]);
  if ( !BYTE4(v41[3]) )
    _libc_free(v41[1]);
}
