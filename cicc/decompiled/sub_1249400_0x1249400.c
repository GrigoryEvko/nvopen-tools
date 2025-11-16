// Function: sub_1249400
// Address: 0x1249400
//
__int64 __fastcall sub_1249400(char *dest, unsigned __int64 a2)
{
  unsigned __int64 v2; // rax
  char *v4; // r13
  void *v5; // rsi
  char *v6; // rdx
  char *v7; // rdi
  char v8; // cl
  unsigned __int64 v9; // [rsp+8h] [rbp-78h] BYREF
  void *src; // [rsp+10h] [rbp-70h] BYREF
  size_t n; // [rsp+18h] [rbp-68h]
  __int64 v12; // [rsp+20h] [rbp-60h]
  char v13; // [rsp+28h] [rbp-58h] BYREF
  char v14[16]; // [rsp+30h] [rbp-50h] BYREF
  unsigned __int64 *v15; // [rsp+40h] [rbp-40h]
  __int16 v16; // [rsp+50h] [rbp-30h]

  v9 = a2;
  if ( a2 <= (unsigned __int64)&loc_98967F )
  {
    n = 0;
    src = &v13;
    v12 = 8;
    v14[0] = 47;
    v15 = &v9;
    v16 = 2824;
    sub_CA0EC0((__int64)v14, (__int64)&src);
    v4 = (char *)src;
    v5 = src;
    memcpy(dest, src, n);
    if ( v4 != &v13 )
      _libc_free(v4, v5);
  }
  else
  {
    v2 = a2;
    if ( a2 > 0xFFFFFFFFFLL )
      return 0;
    *(_WORD *)dest = 12079;
    v6 = dest + 7;
    v7 = dest + 1;
    do
    {
      v8 = v2;
      --v6;
      v2 >>= 6;
      v6[1] = aAbcdefghijklmn_0[v8 & 0x3F];
    }
    while ( v7 != v6 );
  }
  return 1;
}
