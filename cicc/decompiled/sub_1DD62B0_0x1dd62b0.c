// Function: sub_1DD62B0
// Address: 0x1dd62b0
//
__int64 __fastcall sub_1DD62B0(__int64 a1, __int64 a2)
{
  char *v2; // r13
  __int64 v5; // rdi
  __int64 v6; // rdx
  char *v7; // rdi
  size_t v8; // rcx
  __int64 v9; // rdx
  __int64 v10; // rsi
  __int64 v11; // rdi
  const char *v12; // rsi
  unsigned __int64 v13; // rdx
  size_t v15; // rdx
  _QWORD v16[4]; // [rsp+0h] [rbp-80h] BYREF
  char *v17; // [rsp+20h] [rbp-60h] BYREF
  char *v18; // [rsp+28h] [rbp-58h]
  __int16 v19; // [rsp+30h] [rbp-50h]
  char *v20; // [rsp+40h] [rbp-40h] BYREF
  size_t n; // [rsp+48h] [rbp-38h]
  _QWORD src[6]; // [rsp+50h] [rbp-30h] BYREF

  v2 = (char *)(a1 + 16);
  *(_QWORD *)a1 = a1 + 16;
  *(_QWORD *)(a1 + 8) = 0;
  *(_BYTE *)(a1 + 16) = 0;
  v5 = *(_QWORD *)(a2 + 56);
  if ( !v5 )
    goto LABEL_8;
  v16[0] = sub_1E0A440(v5);
  v16[1] = v6;
  v17 = (char *)v16;
  v18 = ":";
  v19 = 773;
  sub_16E2FC0((__int64 *)&v20, (__int64)&v17);
  v7 = *(char **)a1;
  if ( v20 == (char *)src )
  {
    v15 = n;
    if ( n )
    {
      if ( n == 1 )
        *v7 = src[0];
      else
        memcpy(v7, src, n);
      v15 = n;
      v7 = *(char **)a1;
    }
    *(_QWORD *)(a1 + 8) = v15;
    v7[v15] = 0;
    v7 = v20;
    goto LABEL_6;
  }
  v8 = n;
  v9 = src[0];
  if ( v2 == v7 )
  {
    *(_QWORD *)a1 = v20;
    *(_QWORD *)(a1 + 8) = v8;
    *(_QWORD *)(a1 + 16) = v9;
  }
  else
  {
    v10 = *(_QWORD *)(a1 + 16);
    *(_QWORD *)a1 = v20;
    *(_QWORD *)(a1 + 8) = v8;
    *(_QWORD *)(a1 + 16) = v9;
    if ( v7 )
    {
      v20 = v7;
      src[0] = v10;
      goto LABEL_6;
    }
  }
  v20 = (char *)src;
  v7 = (char *)src;
LABEL_6:
  n = 0;
  *v7 = 0;
  if ( v20 != (char *)src )
    j_j___libc_free_0(v20, src[0] + 1LL);
LABEL_8:
  v11 = *(_QWORD *)(a2 + 40);
  if ( v11 )
  {
    v12 = sub_1649960(v11);
    if ( v13 > 0x3FFFFFFFFFFFFFFFLL - *(_QWORD *)(a1 + 8) )
      sub_4262D8((__int64)"basic_string::append");
    sub_2241490(a1, v12);
    return a1;
  }
  LODWORD(v16[0]) = *(_DWORD *)(a2 + 48);
  v17 = "BB";
  v18 = (char *)v16[0];
  v19 = 2563;
  sub_16E2FC0((__int64 *)&v20, (__int64)&v17);
  sub_2241490(a1, v20, n);
  if ( v20 == (char *)src )
    return a1;
  j_j___libc_free_0(v20, src[0] + 1LL);
  return a1;
}
