// Function: sub_BAA900
// Address: 0xbaa900
//
unsigned __int64 __fastcall sub_BAA900(__int64 a1)
{
  _BYTE *v1; // rax
  unsigned __int64 v2; // rax
  __int64 v3; // rsi
  int v4; // edx
  __int64 v5; // r9
  __int64 v6; // r8
  unsigned __int64 v7; // rax

  v1 = (_BYTE *)sub_BA91D0(a1, "SDK Version", 0xBu);
  if ( v1 && *v1 == 1 )
  {
    v2 = sub_BA8450((__int64)v1);
    v3 = (unsigned int)v2;
    v5 = v4 & 0x7FFFFFFF;
    v6 = HIDWORD(v2) & 0x7FFFFFFF;
    v7 = v2 >> 63;
  }
  else
  {
    v5 = 0;
    LOBYTE(v7) = 0;
    v6 = 0;
    v3 = 0;
  }
  return v3 | (((v5 << 32) | v6 | ((unsigned __int64)(unsigned __int8)v7 << 31)) << 32);
}
