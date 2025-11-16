// Function: sub_CC76B0
// Address: 0xcc76b0
//
const char *__fastcall sub_CC76B0(__int64 a1)
{
  void *v1; // rax
  size_t v2; // rdx
  char *v4; // rax
  size_t v5; // rdx
  size_t v6; // r12
  int v7; // edi
  char *v8; // rax
  void *v9; // rdx
  size_t v10; // rbx
  __int64 *v11; // r12
  size_t v12; // r13
  void *s1; // [rsp+0h] [rbp-80h] BYREF
  size_t v14; // [rsp+8h] [rbp-78h]
  void *s2; // [rsp+10h] [rbp-70h] BYREF
  size_t n; // [rsp+18h] [rbp-68h]
  __int64 v17; // [rsp+20h] [rbp-60h] BYREF
  void *v18[4]; // [rsp+30h] [rbp-50h] BYREF
  __int16 v19; // [rsp+50h] [rbp-30h]

  v1 = (void *)sub_CC7490((__int64 *)a1);
  v14 = v2;
  s1 = v1;
  if ( v2 == 4 && *(_DWORD *)s1 == 1701736302 )
    return byte_3F871B3;
  v4 = sub_CC63C0(*(_DWORD *)(a1 + 48));
  v6 = v5;
  if ( v14 >= v5 && (!v5 || !memcmp(s1, v4, v5)) )
  {
    s1 = (char *)s1 + v6;
    v14 -= v6;
  }
  if ( sub_C931B0((__int64 *)&s1, "-", 1u, 0) != -1 )
  {
    v7 = *(_DWORD *)(a1 + 52);
    if ( v7 )
    {
      v8 = sub_CC6710(v7);
      v18[0] = "-";
      v18[2] = v8;
      v18[3] = v9;
      v19 = 1283;
      sub_CA0F50((__int64 *)&s2, v18);
      v10 = v14;
      v11 = (__int64 *)s2;
      if ( n <= v14 )
      {
        v12 = v14 - n;
        if ( !n || !memcmp((char *)s1 + v12, s2, n) )
        {
          if ( v10 > v12 )
            v10 = v12;
          v14 = v10;
        }
      }
      if ( v11 != &v17 )
        j_j___libc_free_0(v11, v17 + 1);
    }
  }
  return (const char *)s1;
}
