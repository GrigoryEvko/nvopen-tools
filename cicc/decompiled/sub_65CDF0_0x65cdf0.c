// Function: sub_65CDF0
// Address: 0x65cdf0
//
__int64 __fastcall sub_65CDF0(char a1, int a2, int *a3, _DWORD *a4, __int64 a5)
{
  __int64 result; // rax
  int v8; // edx
  bool v9; // zf
  _QWORD v10[64]; // [rsp+0h] [rbp-200h] BYREF

  memset(v10, 0, 0x1D8u);
  v10[19] = v10;
  v10[3] = *(_QWORD *)&dword_4F063F8;
  if ( dword_4F077BC && qword_4F077A8 <= 0x9F5Fu )
    BYTE2(v10[22]) |= 1u;
  v10[24] = a5;
  BYTE4(v10[15]) = (4 * (a1 & 1)) | BYTE4(v10[15]) & 0xFB;
  BYTE4(v10[16]) = ((a2 == 0) << 6) | BYTE4(v10[16]) & 0xBF;
  sub_65C7C0((__int64)v10);
  if ( a4 )
    *a4 = (v10[1] >> 5) & 1;
  else
    sub_64EC60((__int64)v10);
  result = v10[36];
  if ( a3 )
  {
    v8 = 0;
    if ( LODWORD(v10[5]) )
    {
      if ( *(_BYTE *)(v10[36] + 140LL) == 12 )
      {
        v9 = (unsigned int)sub_8D4C10(v10[36], 1) == 0;
        result = v10[36];
        v8 = !v9;
      }
    }
    else if ( (v10[15] & 0x7F) != 0 )
    {
      v8 = ((BYTE4(v10[15]) >> 5) ^ 1) & 1;
    }
    *a3 = v8;
  }
  return result;
}
