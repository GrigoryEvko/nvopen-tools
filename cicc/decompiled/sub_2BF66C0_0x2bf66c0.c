// Function: sub_2BF66C0
// Address: 0x2bf66c0
//
void __fastcall sub_2BF66C0(__int64 a1, __int64 *a2)
{
  __int64 v2; // rax
  __int64 v3; // rsi
  __int64 v4; // rdx
  __int64 v5; // rcx
  __int64 v6; // r8
  __int64 v7; // rcx
  __int64 v8; // r8
  int v9; // r12d
  __int64 v10; // rcx
  __int64 v11; // r8
  __int64 v12; // r8
  int v13; // ecx
  __int64 v14; // rcx
  __int64 v15; // r8
  __int64 v16; // r9
  __int64 v17; // rsi
  __m128i *v18; // rdx
  const __m128i *v19; // rax
  unsigned __int64 v20; // rdi
  __int64 v21; // rsi
  __m128i *v22; // rdx
  const __m128i *v23; // rax
  const __m128i *v24; // rdi
  int v25; // [rsp+10h] [rbp-760h]
  __int64 v27; // [rsp+20h] [rbp-750h] BYREF
  __int64 *v28; // [rsp+28h] [rbp-748h]
  int v29; // [rsp+30h] [rbp-740h]
  int v30; // [rsp+34h] [rbp-73Ch]
  int v31; // [rsp+38h] [rbp-738h]
  char v32; // [rsp+3Ch] [rbp-734h]
  __int64 v33; // [rsp+40h] [rbp-730h] BYREF
  const __m128i *v34; // [rsp+80h] [rbp-6F0h]
  __int64 v35; // [rsp+88h] [rbp-6E8h]
  _QWORD v36[24]; // [rsp+90h] [rbp-6E0h] BYREF
  unsigned __int64 v37[38]; // [rsp+150h] [rbp-620h] BYREF
  _BYTE v38[8]; // [rsp+280h] [rbp-4F0h] BYREF
  unsigned __int64 v39; // [rsp+288h] [rbp-4E8h]
  char v40; // [rsp+29Ch] [rbp-4D4h]
  _BYTE v41[64]; // [rsp+2A0h] [rbp-4D0h] BYREF
  __m128i *v42; // [rsp+2E0h] [rbp-490h] BYREF
  __int64 v43; // [rsp+2E8h] [rbp-488h]
  _BYTE v44[192]; // [rsp+2F0h] [rbp-480h] BYREF
  _BYTE v45[8]; // [rsp+3B0h] [rbp-3C0h] BYREF
  unsigned __int64 v46; // [rsp+3B8h] [rbp-3B8h]
  char v47; // [rsp+3CCh] [rbp-3A4h]
  char *v48; // [rsp+410h] [rbp-360h]
  char v49; // [rsp+420h] [rbp-350h] BYREF
  _BYTE v50[8]; // [rsp+4E0h] [rbp-290h] BYREF
  unsigned __int64 v51; // [rsp+4E8h] [rbp-288h]
  char v52; // [rsp+4FCh] [rbp-274h]
  _BYTE v53[64]; // [rsp+500h] [rbp-270h] BYREF
  __m128i *v54; // [rsp+540h] [rbp-230h] BYREF
  __int64 v55; // [rsp+548h] [rbp-228h]
  _BYTE v56[192]; // [rsp+550h] [rbp-220h] BYREF
  _BYTE v57[8]; // [rsp+610h] [rbp-160h] BYREF
  unsigned __int64 v58; // [rsp+618h] [rbp-158h]
  char v59; // [rsp+62Ch] [rbp-144h]
  char *v60; // [rsp+670h] [rbp-100h]
  char v61; // [rsp+680h] [rbp-F0h] BYREF

  memset(v37, 0, sizeof(v37));
  v35 = 0x800000000LL;
  v37[1] = (unsigned __int64)&v37[4];
  v2 = *a2;
  v37[12] = (unsigned __int64)&v37[14];
  HIDWORD(v37[13]) = 8;
  v34 = (const __m128i *)v36;
  v3 = *(unsigned int *)(v2 + 88);
  v28 = &v33;
  v4 = *(_QWORD *)(v2 + 80);
  v33 = v2;
  v36[1] = v4;
  v36[0] = v4 + 8 * v3;
  v36[2] = v2;
  LODWORD(v37[2]) = 8;
  BYTE4(v37[3]) = 1;
  v29 = 8;
  v31 = 0;
  v32 = 1;
  v30 = 1;
  v27 = 1;
  LODWORD(v35) = 1;
  sub_2AD8BC0((__int64)&v27);
  sub_C8CD80((__int64)v50, (__int64)v53, (__int64)v37, v5, v6, (__int64)v50);
  v55 = 0x800000000LL;
  v9 = v37[13];
  v54 = (__m128i *)v56;
  if ( LODWORD(v37[13]) )
  {
    v17 = LODWORD(v37[13]);
    v18 = (__m128i *)v56;
    if ( LODWORD(v37[13]) > 8 )
    {
      sub_2AD8D20((__int64)&v54, LODWORD(v37[13]), (__int64)v56, v7, v8, (__int64)v50);
      v18 = v54;
      v17 = LODWORD(v37[13]);
    }
    v19 = (const __m128i *)v37[12];
    v20 = v37[12] + 24 * v17;
    if ( v37[12] != v20 )
    {
      do
      {
        if ( v18 )
        {
          *v18 = _mm_loadu_si128(v19);
          v18[1].m128i_i64[0] = v19[1].m128i_i64[0];
        }
        v19 = (const __m128i *)((char *)v19 + 24);
        v18 = (__m128i *)((char *)v18 + 24);
      }
      while ( (const __m128i *)v20 != v19 );
    }
    LODWORD(v55) = v9;
  }
  sub_2AD8DC0((__int64)v57, (__int64)v50);
  sub_C8CD80((__int64)v38, (__int64)v41, (__int64)&v27, v10, v11, (__int64)v38);
  v13 = v35;
  v42 = (__m128i *)v44;
  v43 = 0x800000000LL;
  if ( (_DWORD)v35 )
  {
    v21 = (unsigned int)v35;
    v22 = (__m128i *)v44;
    if ( (unsigned int)v35 > 8 )
    {
      v25 = v35;
      sub_2AD8D20((__int64)&v42, (unsigned int)v35, (__int64)v44, (unsigned int)v35, v12, (__int64)v38);
      v22 = v42;
      v21 = (unsigned int)v35;
      v13 = v25;
    }
    v23 = v34;
    v24 = (const __m128i *)((char *)v34 + 24 * v21);
    if ( v34 != v24 )
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
      while ( v24 != v23 );
    }
    LODWORD(v43) = v13;
  }
  sub_2AD8DC0((__int64)v45, (__int64)v38);
  sub_2AD8FA0((__int64)v45, (__int64)v57, a1, v14, v15, v16);
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
  if ( v34 != (const __m128i *)v36 )
    _libc_free((unsigned __int64)v34);
  if ( !v32 )
    _libc_free((unsigned __int64)v28);
  if ( (unsigned __int64 *)v37[12] != &v37[14] )
    _libc_free(v37[12]);
  if ( !BYTE4(v37[3]) )
    _libc_free(v37[1]);
}
