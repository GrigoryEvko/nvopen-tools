// Function: sub_E3EF00
// Address: 0xe3ef00
//
__int64 __fastcall sub_E3EF00(__int64 a1, __int16 a2, __int64 a3, __int64 a4, _BYTE *a5, size_t a6)
{
  size_t v8; // rax
  _QWORD *v9; // rdx
  _QWORD *v10; // rdi
  _BYTE *v11; // rax
  __int64 v12; // rdi
  size_t v13; // rdx
  __int64 *v14; // rdi
  __int64 v15; // rdx
  bool v16; // zf
  __int64 v18; // rax
  _QWORD *v19; // rdi
  _BYTE *src; // [rsp+8h] [rbp-88h]
  size_t v21; // [rsp+18h] [rbp-78h] BYREF
  void *dest; // [rsp+20h] [rbp-70h] BYREF
  size_t v23; // [rsp+28h] [rbp-68h]
  _QWORD v24[2]; // [rsp+30h] [rbp-60h] BYREF
  _QWORD *v25; // [rsp+40h] [rbp-50h] BYREF
  size_t n; // [rsp+48h] [rbp-48h]
  _QWORD v27[8]; // [rsp+50h] [rbp-40h] BYREF

  dest = v24;
  v23 = 0;
  LOBYTE(v24[0]) = 0;
  if ( !a5 )
  {
    LOBYTE(v27[0]) = 0;
    v13 = 0;
    v10 = v24;
    v25 = v27;
LABEL_10:
    v23 = v13;
    *((_BYTE *)v10 + v13) = 0;
    v11 = v25;
    goto LABEL_11;
  }
  v21 = a6;
  v8 = a6;
  v25 = v27;
  if ( a6 > 0xF )
  {
    src = a5;
    v18 = sub_22409D0(&v25, &v21, 0);
    a5 = src;
    v25 = (_QWORD *)v18;
    v19 = (_QWORD *)v18;
    v27[0] = v21;
LABEL_18:
    memcpy(v19, a5, a6);
    v8 = v21;
    v9 = v25;
    goto LABEL_5;
  }
  if ( a6 == 1 )
  {
    LOBYTE(v27[0]) = *a5;
    v9 = v27;
    goto LABEL_5;
  }
  if ( a6 )
  {
    v19 = v27;
    goto LABEL_18;
  }
  v9 = v27;
LABEL_5:
  n = v8;
  *((_BYTE *)v9 + v8) = 0;
  v10 = dest;
  v11 = dest;
  if ( v25 == v27 )
  {
    v13 = n;
    if ( n )
    {
      if ( n == 1 )
        *(_BYTE *)dest = v27[0];
      else
        memcpy(dest, v27, n);
      v13 = n;
      v10 = dest;
    }
    goto LABEL_10;
  }
  if ( dest == v24 )
  {
    dest = v25;
    v23 = n;
    v24[0] = v27[0];
  }
  else
  {
    v12 = v24[0];
    dest = v25;
    v23 = n;
    v24[0] = v27[0];
    if ( v11 )
    {
      v25 = v11;
      v27[0] = v12;
      goto LABEL_11;
    }
  }
  v25 = v27;
  v11 = v27;
LABEL_11:
  n = 0;
  *v11 = 0;
  if ( v25 != v27 )
    j_j___libc_free_0(v25, v27[0] + 1LL);
  v14 = *(__int64 **)(a1 + 136);
  sub_E3ECE0(v14, (__int64 *)&dest);
  v16 = *(_QWORD *)(a1 + 176) == 0;
  *(_WORD *)(a1 + 14) = a2;
  if ( v16 )
    sub_4263D6(v14, &dest, v15);
  (*(void (__fastcall **)(__int64, void **))(a1 + 184))(a1 + 160, &dest);
  if ( dest != v24 )
    j_j___libc_free_0(dest, v24[0] + 1LL);
  return 0;
}
