// Function: sub_87EA80
// Address: 0x87ea80
//
__int128 *sub_87EA80()
{
  __int64 v1; // rdx
  __int64 v2; // rcx
  __int64 v3; // rbx
  char *v4; // rax
  __int64 v5; // rax

  if ( qword_4F600D0 )
    return &xmmword_4F5FE60;
  xmmword_4F5FE60 = (__int128)_mm_load_si128((const __m128i *)&xmmword_4F60040);
  xmmword_4F5FE70 = (__int128)_mm_load_si128((const __m128i *)&xmmword_4F60050);
  xmmword_4F5FE80 = (__int128)_mm_load_si128((const __m128i *)&xmmword_4F60060);
  xmmword_4F5FE90 = (__int128)_mm_load_si128((const __m128i *)&xmmword_4F60070);
  xmmword_4F5FEA0 = (__int128)_mm_load_si128((const __m128i *)&xmmword_4F60080);
  xmmword_4F5FEB0 = (__int128)_mm_load_si128((const __m128i *)&xmmword_4F60090);
  xmmword_4F5FEC0 = (__int128)_mm_load_si128((const __m128i *)&xmmword_4F600A0);
  if ( qword_4F04C68[0] )
  {
    if ( *(int *)(qword_4F04C68[0] + 776LL * dword_4F04C64 + 200) > 0 )
      BYTE5(xmmword_4F5FEB0) |= 1u;
  }
  sub_87E690((__int64)&xmmword_4F5FE60, 8);
  v3 = sub_877070(&xmmword_4F5FE60, 8, v1, v2);
  qword_4F600D0 = v3;
  v4 = (char *)sub_7279A0(10);
  strcpy(v4, "<unnamed>");
  *(_QWORD *)(v3 + 8) = v4;
  v5 = qword_4F600D0;
  *(_BYTE *)(v3 + 73) |= 1u;
  *(_QWORD *)(v3 + 16) = 9;
  *(_QWORD *)&xmmword_4F5FE60 = v5;
  return &xmmword_4F5FE60;
}
