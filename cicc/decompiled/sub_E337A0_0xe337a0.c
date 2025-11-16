// Function: sub_E337A0
// Address: 0xe337a0
//
_BYTE *__fastcall sub_E337A0(unsigned __int64 a1, _WORD *a2)
{
  _BYTE *v2; // rbx
  size_t v3; // rdi
  _BYTE *v4; // rax
  size_t v5; // rdx
  size_t v6; // r13
  size_t v7; // rax
  char v8; // r15
  __int64 v9; // rsi
  _BYTE *v10; // rdi
  __int64 v11; // rsi
  char v13; // bl
  _QWORD v14[3]; // [rsp-98h] [rbp-98h] BYREF
  size_t v15; // [rsp-80h] [rbp-80h]
  _WORD *v16; // [rsp-78h] [rbp-78h]
  __int64 v17; // [rsp-70h] [rbp-70h]
  __int64 v18; // [rsp-68h] [rbp-68h]
  _BYTE *v19; // [rsp-60h] [rbp-60h]
  __int64 v20; // [rsp-58h] [rbp-58h]
  unsigned __int64 v21; // [rsp-50h] [rbp-50h]
  __int64 v22; // [rsp-48h] [rbp-48h]
  __int64 v23; // [rsp-40h] [rbp-40h]

  if ( a1 <= 1 || *a2 != 21087 )
    return 0;
  v2 = a2 + 1;
  v3 = a1 - 2;
  v14[0] = 500;
  v14[1] = 0;
  v14[2] = 0;
  v15 = 0;
  v16 = 0;
  v17 = 0;
  v18 = 1;
  v19 = 0;
  v20 = 0;
  v21 = 0;
  v22 = -1;
  v23 = 1;
  if ( v3 && (v4 = memchr(v2, 46, v3)) != 0 && (v5 = v4 - v2, v6 = v4 - v2, v4 - v2 != -1) )
  {
    v7 = v4 - v2;
    v16 = a2 + 1;
    if ( v3 <= v5 )
      v7 = v3;
    v15 = v7;
    sub_E331B0((__int64)v14, 0, 0);
    if ( v15 != v17 )
    {
      v8 = v18;
      LOBYTE(v18) = 0;
      sub_E331B0((__int64)v14, 0, 0);
      LOBYTE(v18) = v8;
      if ( v17 != v15 )
        BYTE1(v18) = 1;
    }
    sub_E31C60((__int64)v14, 2u, " (");
    if ( v3 < v6 )
      sub_222CF80("%s: __pos (which is %zu) > __size (which is %zu)", (char)"basic_string_view::substr");
    sub_E31C60((__int64)v14, v3 - v6, &v2[v6]);
    v9 = 1;
    sub_E31C60((__int64)v14, 1u, ")");
  }
  else
  {
    v15 = v3;
    v9 = 0;
    v16 = v2;
    sub_E331B0((__int64)v14, 0, 0);
    if ( v17 != v15 )
    {
      v13 = v18;
      v9 = 0;
      LOBYTE(v18) = 0;
      sub_E331B0((__int64)v14, 0, 0);
      v10 = v19;
      LOBYTE(v18) = v13;
      if ( v15 != v17 )
        goto LABEL_24;
    }
  }
  v10 = v19;
  if ( BYTE1(v18) )
  {
LABEL_24:
    _libc_free(v10, v9);
    return 0;
  }
  v11 = v20;
  if ( v20 + 1 > v21 )
  {
    if ( v20 + 993 > 2 * v21 )
      v21 = v20 + 993;
    else
      v21 *= 2LL;
    v19 = (_BYTE *)realloc(v19);
    v10 = v19;
    if ( !v19 )
      abort();
    v11 = v20;
  }
  v10[v11] = 0;
  return v19;
}
