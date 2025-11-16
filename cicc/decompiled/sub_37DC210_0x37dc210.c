// Function: sub_37DC210
// Address: 0x37dc210
//
void __fastcall sub_37DC210(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5)
{
  __int64 *v5; // r9
  __int64 *v6; // rsi
  __int64 v7; // r8
  __int64 v8; // r9
  __int64 v9; // r10
  __int64 v10; // rax
  __int64 v11; // r11
  __int64 v12; // rcx
  int v13; // eax
  __int64 *v14; // rdx
  __int64 v15; // rcx
  __int64 v16; // r9
  __int64 v17; // r9
  int v18; // ecx
  __int64 v19; // rcx
  __int64 v20; // r9
  __int64 v21; // r9
  int v22; // ecx
  __int64 v23; // rcx
  __int64 v24; // r8
  __int64 v25; // r9
  __int64 v26; // rsi
  __m128i *v27; // rdx
  const __m128i *v28; // rax
  unsigned __int64 v29; // rdi
  __int64 v30; // rsi
  __m128i *v31; // rdx
  const __m128i *v32; // rax
  const __m128i *v33; // rdi
  __int64 v34; // rdi
  _QWORD *v35; // rax
  const __m128i *v36; // rax
  const __m128i *v37; // rdi
  int v38; // eax
  __int64 v39; // [rsp+0h] [rbp-770h]
  __int64 v40; // [rsp+8h] [rbp-768h]
  int v41; // [rsp+8h] [rbp-768h]
  __int64 v42; // [rsp+10h] [rbp-760h]
  int v43; // [rsp+10h] [rbp-760h]
  __int64 v44; // [rsp+10h] [rbp-760h]
  int v45; // [rsp+10h] [rbp-760h]
  __int64 v47; // [rsp+20h] [rbp-750h] BYREF
  char *v48; // [rsp+28h] [rbp-748h]
  __int64 v49; // [rsp+30h] [rbp-740h]
  int v50; // [rsp+38h] [rbp-738h]
  char v51; // [rsp+3Ch] [rbp-734h]
  char v52; // [rsp+40h] [rbp-730h] BYREF
  const __m128i *v53; // [rsp+80h] [rbp-6F0h] BYREF
  __int64 v54; // [rsp+88h] [rbp-6E8h]
  _BYTE v55[192]; // [rsp+90h] [rbp-6E0h] BYREF
  unsigned __int64 v56[38]; // [rsp+150h] [rbp-620h] BYREF
  char v57[8]; // [rsp+280h] [rbp-4F0h] BYREF
  unsigned __int64 v58; // [rsp+288h] [rbp-4E8h]
  char v59; // [rsp+29Ch] [rbp-4D4h]
  char v60[64]; // [rsp+2A0h] [rbp-4D0h] BYREF
  __m128i *v61; // [rsp+2E0h] [rbp-490h] BYREF
  __int64 v62; // [rsp+2E8h] [rbp-488h]
  _BYTE v63[192]; // [rsp+2F0h] [rbp-480h] BYREF
  char v64[8]; // [rsp+3B0h] [rbp-3C0h] BYREF
  unsigned __int64 v65; // [rsp+3B8h] [rbp-3B8h]
  char v66; // [rsp+3CCh] [rbp-3A4h]
  char *v67; // [rsp+410h] [rbp-360h]
  char v68; // [rsp+420h] [rbp-350h] BYREF
  char v69[8]; // [rsp+4E0h] [rbp-290h] BYREF
  unsigned __int64 v70; // [rsp+4E8h] [rbp-288h]
  char v71; // [rsp+4FCh] [rbp-274h]
  char v72[64]; // [rsp+500h] [rbp-270h] BYREF
  __m128i *v73; // [rsp+540h] [rbp-230h] BYREF
  __int64 v74; // [rsp+548h] [rbp-228h]
  _BYTE v75[192]; // [rsp+550h] [rbp-220h] BYREF
  unsigned __int64 v76[3]; // [rsp+610h] [rbp-160h] BYREF
  char v77; // [rsp+62Ch] [rbp-144h]
  char *v78; // [rsp+670h] [rbp-100h]
  char v79; // [rsp+680h] [rbp-F0h] BYREF

  v5 = *(__int64 **)(a2 + 328);
  memset(v56, 0, sizeof(v56));
  v6 = &v47;
  v56[1] = (unsigned __int64)&v56[4];
  v48 = &v52;
  v54 = 0x800000000LL;
  v42 = (__int64)v5;
  LODWORD(v56[2]) = 8;
  BYTE4(v56[3]) = 1;
  v56[12] = (unsigned __int64)&v56[14];
  HIDWORD(v56[13]) = 8;
  v47 = 0;
  v49 = 8;
  v50 = 0;
  v51 = 1;
  v53 = (const __m128i *)v55;
  sub_37BC2F0((__int64)v76, (__int64)&v47, v5, 0, a5, (__int64)v5);
  v8 = v42;
  v9 = *(_QWORD *)(v42 + 112);
  v10 = *(unsigned int *)(v42 + 120);
  v11 = v9 + 8 * v10;
  if ( HIDWORD(v54) <= (unsigned int)v54 )
  {
    v6 = (__int64 *)v55;
    v39 = v9 + 8 * v10;
    v40 = *(_QWORD *)(v42 + 112);
    v12 = sub_C8D7D0((__int64)&v53, (__int64)v55, 0, 0x18u, v76, v42);
    v34 = 24LL * (unsigned int)v54;
    v35 = (_QWORD *)(v34 + v12);
    if ( v34 + v12 )
    {
      v8 = v42;
      *v35 = v39;
      v35[1] = v40;
      v35[2] = v42;
      v34 = 24LL * (unsigned int)v54;
    }
    v36 = v53;
    v37 = (const __m128i *)((char *)v53 + v34);
    if ( v53 != v37 )
    {
      v14 = (__int64 *)v12;
      do
      {
        if ( v14 )
        {
          *v14 = v36->m128i_i64[0];
          v14[1] = v36->m128i_i64[1];
          v6 = (__int64 *)v36[1].m128i_i64[0];
          v14[2] = (__int64)v6;
        }
        v36 = (const __m128i *)((char *)v36 + 24);
        v14 += 3;
      }
      while ( v37 != v36 );
      v37 = v53;
    }
    v38 = v76[0];
    if ( v37 != (const __m128i *)v55 )
    {
      v41 = v76[0];
      v44 = v12;
      _libc_free((unsigned __int64)v37);
      v38 = v41;
      v12 = v44;
    }
    LODWORD(v54) = v54 + 1;
    v53 = (const __m128i *)v12;
    HIDWORD(v54) = v38;
  }
  else
  {
    v12 = 3LL * (unsigned int)v54;
    v13 = v54;
    v14 = &v53->m128i_i64[3 * (unsigned int)v54];
    if ( v14 )
    {
      *v14 = v11;
      v14[1] = v9;
      v14[2] = v42;
      v13 = v54;
    }
    LODWORD(v54) = v13 + 1;
  }
  sub_2DACB60((__int64)&v47, (__int64)v6, (__int64)v14, v12, v7, v8);
  sub_C8CD80((__int64)v69, (__int64)v72, (__int64)v56, v15, (__int64)v69, v16);
  v18 = v56[13];
  v73 = (__m128i *)v75;
  v74 = 0x800000000LL;
  if ( LODWORD(v56[13]) )
  {
    v26 = LODWORD(v56[13]);
    v27 = (__m128i *)v75;
    if ( LODWORD(v56[13]) > 8 )
    {
      v43 = v56[13];
      sub_2DACD40((__int64)&v73, LODWORD(v56[13]), (__int64)v75, LODWORD(v56[13]), (__int64)v69, v17);
      v27 = v73;
      v26 = LODWORD(v56[13]);
      v18 = v43;
    }
    v28 = (const __m128i *)v56[12];
    v29 = v56[12] + 24 * v26;
    if ( v56[12] != v29 )
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
      while ( (const __m128i *)v29 != v28 );
    }
    LODWORD(v74) = v18;
  }
  sub_2DACDE0((__int64)v76, (__int64)v69);
  sub_C8CD80((__int64)v57, (__int64)v60, (__int64)&v47, v19, (__int64)v57, v20);
  v22 = v54;
  v61 = (__m128i *)v63;
  v62 = 0x800000000LL;
  if ( (_DWORD)v54 )
  {
    v30 = (unsigned int)v54;
    v31 = (__m128i *)v63;
    if ( (unsigned int)v54 > 8 )
    {
      v45 = v54;
      sub_2DACD40((__int64)&v61, (unsigned int)v54, (__int64)v63, (unsigned int)v54, (__int64)v57, v21);
      v31 = v61;
      v30 = (unsigned int)v54;
      v22 = v45;
    }
    v32 = v53;
    v33 = (const __m128i *)((char *)v53 + 24 * v30);
    if ( v53 != v33 )
    {
      do
      {
        if ( v31 )
        {
          *v31 = _mm_loadu_si128(v32);
          v31[1].m128i_i64[0] = v32[1].m128i_i64[0];
        }
        v32 = (const __m128i *)((char *)v32 + 24);
        v31 = (__m128i *)((char *)v31 + 24);
      }
      while ( v33 != v32 );
    }
    LODWORD(v62) = v22;
  }
  sub_2DACDE0((__int64)v64, (__int64)v57);
  sub_2E564A0((__int64)v64, (__int64)v76, a1, v23, v24, v25);
  if ( v67 != &v68 )
    _libc_free((unsigned __int64)v67);
  if ( !v66 )
    _libc_free(v65);
  if ( v61 != (__m128i *)v63 )
    _libc_free((unsigned __int64)v61);
  if ( !v59 )
    _libc_free(v58);
  if ( v78 != &v79 )
    _libc_free((unsigned __int64)v78);
  if ( !v77 )
    _libc_free(v76[1]);
  if ( v73 != (__m128i *)v75 )
    _libc_free((unsigned __int64)v73);
  if ( !v71 )
    _libc_free(v70);
  if ( v53 != (const __m128i *)v55 )
    _libc_free((unsigned __int64)v53);
  if ( !v51 )
    _libc_free((unsigned __int64)v48);
  if ( (unsigned __int64 *)v56[12] != &v56[14] )
    _libc_free(v56[12]);
  if ( !BYTE4(v56[3]) )
    _libc_free(v56[1]);
}
