// Function: sub_CC7810
// Address: 0xcc7810
//
unsigned __int64 __fastcall sub_CC7810(__int64 a1)
{
  const char *v1; // rax
  __int64 v2; // rdx
  unsigned __int8 v3; // dl
  unsigned int v4; // r10d
  __int64 v5; // rdi
  __int64 v6; // r9
  __int128 v8; // [rsp+0h] [rbp-10h] BYREF

  v1 = sub_CC76B0(a1);
  v8 = 0;
  sub_F05080(&v8, v1, v2);
  if ( v8 < 0 )
  {
    v5 = (unsigned int)v8;
    v3 = 1;
    v6 = DWORD2(v8) & 0x7FFFFFFF;
    v4 = DWORD1(v8) & 0x7FFFFFFF;
  }
  else
  {
    v3 = BYTE7(v8) >> 7;
    v4 = DWORD1(v8) & 0x7FFFFFFF;
    v5 = (unsigned int)v8;
    v6 = DWORD2(v8) & 0x7FFFFFFF;
  }
  return v5 | (((v6 << 32) | v4 | ((unsigned __int64)v3 << 31)) << 32);
}
