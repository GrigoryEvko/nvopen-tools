// Function: sub_2C40060
// Address: 0x2c40060
//
void __fastcall sub_2C40060(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  __int64 v6; // r9
  __int64 v7; // r15
  __int64 v8; // rax
  __int64 v9; // r11
  int v10; // eax
  __int64 *v11; // rdx
  __int64 v12; // rcx
  __int64 v13; // r8
  __int64 v14; // r9
  __int64 v15; // rdx
  __int64 v16; // r8
  __int64 v17; // r9
  int v18; // r15d
  __m128i *v19; // rax
  __int64 v20; // rcx
  __int64 v21; // r8
  __int64 v22; // r9
  __int64 v23; // rdx
  __int64 v24; // r8
  __int64 v25; // r9
  __m128i *v26; // rax
  int v27; // ecx
  __int64 v28; // rcx
  __int64 v29; // r8
  __int64 v30; // r9
  __int64 v31; // rsi
  const __m128i *v32; // rdx
  __int64 v33; // rdi
  __int64 v34; // rsi
  const __m128i *v35; // rdx
  const __m128i *v36; // rdi
  __int64 v37; // rcx
  __int64 v38; // rdi
  _QWORD *v39; // rax
  const __m128i *v40; // rax
  const __m128i *v41; // rdi
  _QWORD *v42; // rdx
  int v43; // ebx
  __int64 v44; // [rsp+8h] [rbp-768h]
  __int64 v45; // [rsp+10h] [rbp-760h]
  int v46; // [rsp+10h] [rbp-760h]
  _QWORD v48[3]; // [rsp+20h] [rbp-750h] BYREF
  int v49; // [rsp+38h] [rbp-738h]
  char v50; // [rsp+3Ch] [rbp-734h]
  char v51; // [rsp+40h] [rbp-730h] BYREF
  const __m128i *v52; // [rsp+80h] [rbp-6F0h] BYREF
  __int64 v53; // [rsp+88h] [rbp-6E8h]
  _BYTE v54[192]; // [rsp+90h] [rbp-6E0h] BYREF
  _QWORD v55[38]; // [rsp+150h] [rbp-620h] BYREF
  _BYTE v56[32]; // [rsp+280h] [rbp-4F0h] BYREF
  char v57[64]; // [rsp+2A0h] [rbp-4D0h] BYREF
  __m128i *v58; // [rsp+2E0h] [rbp-490h] BYREF
  __int64 v59; // [rsp+2E8h] [rbp-488h]
  _BYTE v60[192]; // [rsp+2F0h] [rbp-480h] BYREF
  _BYTE v61[304]; // [rsp+3B0h] [rbp-3C0h] BYREF
  _BYTE v62[32]; // [rsp+4E0h] [rbp-290h] BYREF
  char v63[64]; // [rsp+500h] [rbp-270h] BYREF
  __m128i *v64; // [rsp+540h] [rbp-230h] BYREF
  __int64 v65; // [rsp+548h] [rbp-228h]
  _BYTE v66[192]; // [rsp+550h] [rbp-220h] BYREF
  unsigned __int64 v67[44]; // [rsp+610h] [rbp-160h] BYREF

  memset(v55, 0, sizeof(v55));
  v52 = (const __m128i *)v54;
  v55[1] = &v55[4];
  v55[12] = &v55[14];
  v48[1] = &v51;
  v53 = 0x800000000LL;
  LODWORD(v55[2]) = 8;
  BYTE4(v55[3]) = 1;
  HIDWORD(v55[13]) = 8;
  v48[0] = 0;
  v48[2] = 8;
  v49 = 0;
  v50 = 1;
  sub_2C3F110((__int64)v67, (__int64)v48, (__int64 *)a2, 0, a5, a6);
  v7 = *(_QWORD *)(a2 + 80);
  v8 = *(unsigned int *)(a2 + 88);
  v9 = v7 + 8 * v8;
  if ( HIDWORD(v53) <= (unsigned int)v53 )
  {
    v44 = v7 + 8 * v8;
    v37 = sub_C8D7D0((__int64)&v52, (__int64)v54, 0, 0x18u, v67, v6);
    v38 = 24LL * (unsigned int)v53;
    v39 = (_QWORD *)(v38 + v37);
    if ( v38 + v37 )
    {
      v39[1] = v7;
      v39[2] = a2;
      *v39 = v44;
      v38 = 24LL * (unsigned int)v53;
    }
    v40 = v52;
    v41 = (const __m128i *)((char *)v52 + v38);
    if ( v52 != v41 )
    {
      v42 = (_QWORD *)v37;
      do
      {
        if ( v42 )
        {
          *v42 = v40->m128i_i64[0];
          v42[1] = v40->m128i_i64[1];
          v42[2] = v40[1].m128i_i64[0];
        }
        v40 = (const __m128i *)((char *)v40 + 24);
        v42 += 3;
      }
      while ( v41 != v40 );
      v41 = v52;
    }
    v43 = v67[0];
    if ( v41 != (const __m128i *)v54 )
    {
      v45 = v37;
      _libc_free((unsigned __int64)v41);
      v37 = v45;
    }
    LODWORD(v53) = v53 + 1;
    v52 = (const __m128i *)v37;
    HIDWORD(v53) = v43;
  }
  else
  {
    v10 = v53;
    v11 = &v52->m128i_i64[3 * (unsigned int)v53];
    if ( v11 )
    {
      *v11 = v9;
      v11[1] = v7;
      v11[2] = a2;
      v10 = v53;
    }
    LODWORD(v53) = v10 + 1;
  }
  sub_2AD8BC0((__int64)v48);
  sub_C8CD80((__int64)v62, (__int64)v63, (__int64)v55, v12, v13, v14);
  v18 = v55[13];
  v19 = (__m128i *)v66;
  v64 = (__m128i *)v66;
  v65 = 0x800000000LL;
  if ( LODWORD(v55[13]) )
  {
    v31 = LODWORD(v55[13]);
    if ( LODWORD(v55[13]) > 8 )
    {
      sub_2AD8D20((__int64)&v64, LODWORD(v55[13]), v15, 0x800000000LL, v16, v17);
      v19 = v64;
      v31 = LODWORD(v55[13]);
    }
    v32 = (const __m128i *)v55[12];
    v33 = v55[12] + 24 * v31;
    if ( v55[12] != v33 )
    {
      do
      {
        if ( v19 )
        {
          *v19 = _mm_loadu_si128(v32);
          v19[1].m128i_i64[0] = v32[1].m128i_i64[0];
        }
        v32 = (const __m128i *)((char *)v32 + 24);
        v19 = (__m128i *)((char *)v19 + 24);
      }
      while ( (const __m128i *)v33 != v32 );
    }
    LODWORD(v65) = v18;
  }
  sub_2AD8DC0((__int64)v67, (__int64)v62);
  sub_C8CD80((__int64)v56, (__int64)v57, (__int64)v48, v20, v21, v22);
  v26 = (__m128i *)v60;
  v59 = 0x800000000LL;
  v27 = v53;
  v58 = (__m128i *)v60;
  if ( (_DWORD)v53 )
  {
    v34 = (unsigned int)v53;
    if ( (unsigned int)v53 > 8 )
    {
      v46 = v53;
      sub_2AD8D20((__int64)&v58, (unsigned int)v53, v23, (unsigned int)v53, v24, v25);
      v26 = v58;
      v34 = (unsigned int)v53;
      v27 = v46;
    }
    v35 = v52;
    v36 = (const __m128i *)((char *)v52 + 24 * v34);
    if ( v52 != v36 )
    {
      do
      {
        if ( v26 )
        {
          *v26 = _mm_loadu_si128(v35);
          v26[1].m128i_i64[0] = v35[1].m128i_i64[0];
        }
        v35 = (const __m128i *)((char *)v35 + 24);
        v26 = (__m128i *)((char *)v26 + 24);
      }
      while ( v36 != v35 );
    }
    LODWORD(v59) = v27;
  }
  sub_2AD8DC0((__int64)v61, (__int64)v56);
  sub_2AD8FA0((__int64)v61, (__int64)v67, a1, v28, v29, v30);
  sub_2AC2230((__int64)v61);
  sub_2AC2230((__int64)v56);
  sub_2AC2230((__int64)v67);
  sub_2AC2230((__int64)v62);
  sub_2AC2230((__int64)v48);
  sub_2AC2230((__int64)v55);
}
