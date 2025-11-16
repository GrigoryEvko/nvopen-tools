// Function: sub_8F9010
// Address: 0x8f9010
//
_DWORD *sub_8F9010()
{
  char *v0; // rax
  __int64 v1; // rdx
  __int64 v2; // rcx
  __m128i *v3; // rbx
  __m128i v4; // xmm1
  __int32 v5; // eax
  __m128i *v6; // rbx
  __m128i v7; // xmm5
  __int32 v8; // eax
  __m128i *v9; // rbx
  __m128i v10; // xmm1
  __int32 v11; // eax
  __m128i *v12; // rbx
  __m128i v13; // xmm1
  __int32 v14; // eax
  __m128i *v15; // rbx
  const __m128i *v16; // rax
  __m128i *v17; // rbx
  __m128i v18; // xmm7
  __int32 v19; // eax
  __m128i *v20; // rbx
  __m128i v21; // xmm7
  __int64 v22; // rdi
  __m128i *v23; // rbx
  __m128i v24; // xmm5
  __int32 v25; // eax
  __m128i *v26; // rbx
  __m128i v27; // xmm5
  __int64 v28; // rdi
  __m128i *v29; // rbx
  __m128i v30; // xmm3
  __int32 v31; // eax
  __m128i *v32; // rbx
  __m128i v33; // xmm7
  __int64 v34; // rdi
  __m128i *v36; // rbx
  __m128i v37; // xmm7
  __int32 v38; // eax
  __m128i *v39; // rbx
  __m128i v40; // xmm1
  __int64 v41; // rdi

  v0 = (char *)&unk_4F69100;
  v1 = qword_4F690E0;
  do
  {
    v2 = v1;
    v1 = (__int64)v0;
    v0 += 2096;
    *((_QWORD *)v0 - 262) = v2;
  }
  while ( &byte_4F6D280 != v0 );
  dword_4F62E90 = 0;
  v3 = (__m128i *)&unk_4F62520;
  qword_4F690E0 = (__int64)&unk_4F6CA50;
  dword_4B7F248 = 9;
  qword_4F62500[0] = 6;
  dword_4F6251C = 24;
  sub_8F0790((__int64)qword_4F62500, 1u);
  do
  {
    v4 = _mm_loadu_si128(v3 - 1);
    v5 = v3[-2].m128i_i32[0];
    *v3 = _mm_loadu_si128(v3 - 2);
    v3[1] = v4;
    if ( v5 != 6 )
      sub_8F06E0(v3, 10);
    v3 += 2;
  }
  while ( &unk_4F62660 != (_UNKNOWN *)v3 );
  v6 = (__m128i *)&unk_4F62360;
  xmmword_4F624A0[0] = (__int128)_mm_load_si128((const __m128i *)&xmmword_4F62600);
  xmmword_4F624B0 = (__int128)_mm_load_si128((const __m128i *)&xmmword_4F62610);
  xmmword_4F624C0 = xmmword_4F624A0[0];
  xmmword_4F624D0 = xmmword_4F624B0;
  sub_8EF960(&xmmword_4F624C0, &xmmword_4F624C0);
  dword_4B7F248 = 2 * dword_4B7F248 - 1;
  sub_8F1B70((__int64)&xmmword_4F624C0, (__int64)&unk_4B7F240, 0);
  xmmword_4F624E0 = (__int128)_mm_load_si128((const __m128i *)&xmmword_4F624C0);
  xmmword_4F624F0 = (__int128)_mm_load_si128((const __m128i *)&xmmword_4F624D0);
  sub_8EF960(&xmmword_4F624E0, &xmmword_4F624E0);
  dword_4B7F248 = 2 * dword_4B7F248 - 1;
  sub_8F1B70((__int64)&xmmword_4F624E0, (__int64)&unk_4B7F240, 0);
  dword_4B7F208 = 9;
  qword_4F62340 = 6;
  dword_4F6235C = 24;
  sub_8F0790((__int64)&qword_4F62340, 1u);
  do
  {
    v7 = _mm_loadu_si128(v6 - 1);
    v8 = v6[-2].m128i_i32[0];
    *v6 = _mm_loadu_si128(v6 - 2);
    v6[1] = v7;
    if ( v8 != 6 )
      sub_8F06E0(v6, 10);
    v6 += 2;
  }
  while ( xmmword_4F624A0 != (__int128 *)v6 );
  v9 = (__m128i *)&unk_4F621A0;
  xmmword_4F622E0 = (__int128)_mm_load_si128((const __m128i *)&xmmword_4F62440);
  xmmword_4F622F0 = (__int128)_mm_load_si128(xmmword_4F62450);
  xmmword_4F62300 = xmmword_4F622E0;
  xmmword_4F62310 = xmmword_4F622F0;
  sub_8EF960(&xmmword_4F62300, &xmmword_4F62300);
  dword_4B7F208 = 2 * dword_4B7F208 - 1;
  sub_8F1B70((__int64)&xmmword_4F62300, (__int64)&unk_4B7F200, 0);
  xmmword_4F62320 = (__int128)_mm_load_si128((const __m128i *)&xmmword_4F62300);
  xmmword_4F62330 = (__int128)_mm_load_si128((const __m128i *)&xmmword_4F62310);
  sub_8EF960(&xmmword_4F62320, &xmmword_4F62320);
  dword_4B7F208 = 2 * dword_4B7F208 - 1;
  sub_8F1B70((__int64)&xmmword_4F62320, (__int64)&unk_4B7F200, 0);
  dword_4B7F1C8 = 9;
  qword_4F62180[0] = 6;
  dword_4F6219C = 24;
  sub_8F0790((__int64)qword_4F62180, 1u);
  do
  {
    v10 = _mm_loadu_si128(v9 - 1);
    v11 = v9[-2].m128i_i32[0];
    *v9 = _mm_loadu_si128(v9 - 2);
    v9[1] = v10;
    if ( v11 != 6 )
      sub_8F06E0(v9, 10);
    v9 += 2;
  }
  while ( &xmmword_4F622E0 != (__int128 *)v9 );
  v12 = (__m128i *)&unk_4F61E60;
  xmmword_4F62120[0] = (__int128)_mm_load_si128((const __m128i *)&xmmword_4F62280);
  xmmword_4F62130 = (__int128)_mm_load_si128((const __m128i *)&xmmword_4F62290);
  xmmword_4F62140 = xmmword_4F62120[0];
  xmmword_4F62150 = xmmword_4F62130;
  sub_8EF960(&xmmword_4F62140, &xmmword_4F62140);
  dword_4B7F1C8 = 2 * dword_4B7F1C8 - 1;
  sub_8F1B70((__int64)&xmmword_4F62140, (__int64)&unk_4B7F1C0, 0);
  xmmword_4F62160 = (__int128)_mm_load_si128((const __m128i *)&xmmword_4F62140);
  xmmword_4F62170 = (__int128)_mm_load_si128((const __m128i *)&xmmword_4F62150);
  sub_8EF960(&xmmword_4F62160, &xmmword_4F62160);
  dword_4B7F1C8 = 2 * dword_4B7F1C8 - 1;
  sub_8F1B70((__int64)&xmmword_4F62160, (__int64)&unk_4B7F1C0, 0);
  dword_4B7F188 = 17;
  qword_4F61E40[0] = 6;
  dword_4F61E5C = 53;
  sub_8F0790((__int64)qword_4F61E40, 1u);
  do
  {
    v13 = _mm_loadu_si128(v12 - 1);
    v14 = v12[-2].m128i_i32[0];
    *v12 = _mm_loadu_si128(v12 - 2);
    v12[1] = v13;
    if ( v14 != 6 )
      sub_8F06E0(v12, 10);
    v12 += 2;
  }
  while ( xmmword_4F62120 != (__int128 *)v12 );
  v15 = (__m128i *)xmmword_4F61DA0;
  xmmword_4F61DA0[0] = (__int128)_mm_load_si128((const __m128i *)&xmmword_4F62040);
  xmmword_4F61DB0 = (__int128)_mm_load_si128(xmmword_4F62050);
  do
  {
    v16 = v15;
    v15 += 2;
    *v15 = _mm_load_si128(v16);
    v15[1] = _mm_load_si128(v16 + 1);
    sub_8EF960(v15, v15);
    dword_4B7F188 = 2 * dword_4B7F188 - 1;
    sub_8F1B70((__int64)v15, (__int64)&unk_4B7F180, 0);
  }
  while ( &unk_4F61E20 != (_UNKNOWN *)v15 );
  if ( unk_4D04520 )
  {
    v17 = (__m128i *)&unk_4F61300;
    dword_4B7F108 = 17;
    qword_4F612E0[0] = 6;
    dword_4F612FC = 113;
    sub_8F0790((__int64)qword_4F612E0, 1u);
    do
    {
      v18 = _mm_loadu_si128(v17 - 1);
      v19 = v17[-2].m128i_i32[0];
      *v17 = _mm_loadu_si128(v17 - 2);
      v17[1] = v18;
      if ( v19 != 6 )
        sub_8F06E0(v17, 10);
      v17 += 2;
    }
    while ( xmmword_4F61900 != (__int128 *)v17 );
    v20 = (__m128i *)&unk_4F611E0;
    xmmword_4F611C0[0] = (__int128)_mm_load_si128((const __m128i *)&xmmword_4F614E0);
    xmmword_4F611D0 = (__int128)_mm_load_si128(xmmword_4F614F0);
    do
    {
      v21 = _mm_loadu_si128(v20 - 1);
      *v20 = _mm_loadu_si128(v20 - 2);
      v20[1] = v21;
      sub_8EF960(v20, v20);
      v22 = (__int64)v20;
      v20 += 2;
      dword_4B7F108 = 2 * dword_4B7F108 - 1;
      sub_8F1B70(v22, (__int64)&unk_4B7F100, 0);
    }
    while ( qword_4F612E0 != (__int64 *)v20 );
  }
  else
  {
    v36 = (__m128i *)&unk_4F61A40;
    dword_4B7F148 = 17;
    qword_4F61A20[0] = 6;
    dword_4F61A3C = 64;
    sub_8F0790((__int64)qword_4F61A20, 1u);
    do
    {
      v37 = _mm_loadu_si128(v36 - 1);
      v38 = v36[-2].m128i_i32[0];
      *v36 = _mm_loadu_si128(v36 - 2);
      v36[1] = v37;
      if ( v38 != 6 )
        sub_8F06E0(v36, 10);
      v36 += 2;
    }
    while ( xmmword_4F61DA0 != (__int128 *)v36 );
    v39 = (__m128i *)&unk_4F61920;
    xmmword_4F61900[0] = (__int128)_mm_load_si128((const __m128i *)&xmmword_4F61C20);
    xmmword_4F61910 = (__int128)_mm_load_si128(xmmword_4F61C30);
    do
    {
      v40 = _mm_loadu_si128(v39 - 1);
      *v39 = _mm_loadu_si128(v39 - 2);
      v39[1] = v40;
      sub_8EF960(v39, v39);
      v41 = (__int64)v39;
      v39 += 2;
      dword_4B7F148 = 2 * dword_4B7F148 - 1;
      sub_8F1B70(v41, (__int64)&unk_4B7F140, 0);
    }
    while ( qword_4F61A20 != (__int64 *)v39 );
  }
  v23 = (__m128i *)&unk_4F60E60;
  dword_4B7F0C8 = 17;
  qword_4F60E40[0] = 6;
  dword_4F60E5C = 64;
  sub_8F0790((__int64)qword_4F60E40, 1u);
  do
  {
    v24 = _mm_loadu_si128(v23 - 1);
    v25 = v23[-2].m128i_i32[0];
    *v23 = _mm_loadu_si128(v23 - 2);
    v23[1] = v24;
    if ( v25 != 6 )
      sub_8F06E0(v23, 10);
    v23 += 2;
  }
  while ( xmmword_4F611C0 != (__int128 *)v23 );
  v26 = (__m128i *)&unk_4F60D40;
  xmmword_4F60D20[0] = (__int128)_mm_load_si128((const __m128i *)&xmmword_4F61040);
  xmmword_4F60D30 = (__int128)_mm_load_si128(xmmword_4F61050);
  do
  {
    v27 = _mm_loadu_si128(v26 - 1);
    *v26 = _mm_loadu_si128(v26 - 2);
    v26[1] = v27;
    sub_8EF960(v26, v26);
    v28 = (__int64)v26;
    v26 += 2;
    dword_4B7F0C8 = 2 * dword_4B7F0C8 - 1;
    sub_8F1B70(v28, (__int64)&unk_4B7F0C0, 0);
  }
  while ( qword_4F60E40 != (__int64 *)v26 );
  v29 = (__m128i *)&unk_4F60720;
  dword_4B7F088 = 17;
  qword_4F60700[0] = 6;
  dword_4F6071C = 113;
  sub_8F0790((__int64)qword_4F60700, 1u);
  do
  {
    v30 = _mm_loadu_si128(v29 - 1);
    v31 = v29[-2].m128i_i32[0];
    *v29 = _mm_loadu_si128(v29 - 2);
    v29[1] = v30;
    if ( v31 != 6 )
      sub_8F06E0(v29, 10);
    v29 += 2;
  }
  while ( xmmword_4F60D20 != (__int128 *)v29 );
  v32 = (__m128i *)&unk_4F60600;
  xmmword_4F605E0[0] = (__int128)_mm_load_si128((const __m128i *)&xmmword_4F60900);
  xmmword_4F605F0 = (__int128)_mm_load_si128(xmmword_4F60910);
  do
  {
    v33 = _mm_loadu_si128(v32 - 1);
    *v32 = _mm_loadu_si128(v32 - 2);
    v32[1] = v33;
    sub_8EF960(v32, v32);
    v34 = (__int64)v32;
    v32 += 2;
    dword_4B7F088 = 2 * dword_4B7F088 - 1;
    sub_8F1B70(v34, (__int64)&unk_4B7F080, 0);
  }
  while ( v32 != (__m128i *)qword_4F60700 );
  dword_4F066B4[0] = 1;
  return dword_4F066B4;
}
