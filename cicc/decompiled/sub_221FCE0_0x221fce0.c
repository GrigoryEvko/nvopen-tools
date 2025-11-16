// Function: sub_221FCE0
// Address: 0x221fce0
//
__int64 __fastcall sub_221FCE0(
        __int64 a1,
        __int64 a2,
        __int64 a3,
        __int64 a4,
        __int64 a5,
        unsigned __int8 a6,
        __int64 a7,
        _DWORD *a8,
        _QWORD *a9)
{
  __int64 *v9; // rdi
  __int64 v10; // r13
  _QWORD *v12; // rdi
  size_t v13; // rdx
  __int64 v14; // rcx
  __int64 v15; // rsi
  int v16; // [rsp+1Ch] [rbp-84h] BYREF
  _QWORD *v17; // [rsp+20h] [rbp-80h] BYREF
  size_t n; // [rsp+28h] [rbp-78h]
  _QWORD src[2]; // [rsp+30h] [rbp-70h] BYREF
  _BYTE *v20[4]; // [rsp+40h] [rbp-60h] BYREF
  void (__fastcall *v21)(_BYTE **); // [rsp+60h] [rbp-40h]

  v9 = *(__int64 **)(a1 + 16);
  v21 = 0;
  v16 = 0;
  v10 = sub_2214360(v9, a2, a3, a4, a5, a6, a7, &v16, 0, (__int64)v20);
  if ( v16 )
  {
    *a8 = v16;
    goto LABEL_3;
  }
  if ( !v21 )
    sub_426248((__int64)"uninitialized __any_string");
  v17 = src;
  sub_221FC40((__int64 *)&v17, v20[0], (__int64)&v20[0][(unsigned __int64)v20[1]]);
  v12 = (_QWORD *)*a9;
  v13 = n;
  if ( v17 == src )
  {
    if ( n )
    {
      if ( n == 1 )
        *(_BYTE *)v12 = src[0];
      else
        memcpy(v12, src, n);
      v13 = n;
      v12 = (_QWORD *)*a9;
    }
    a9[1] = v13;
    *((_BYTE *)v12 + v13) = 0;
    v12 = v17;
    goto LABEL_11;
  }
  v14 = src[0];
  if ( v12 == a9 + 2 )
  {
    *a9 = v17;
    a9[1] = v13;
    a9[2] = v14;
  }
  else
  {
    v15 = a9[2];
    *a9 = v17;
    a9[1] = v13;
    a9[2] = v14;
    if ( v12 )
    {
      v17 = v12;
      src[0] = v15;
      goto LABEL_11;
    }
  }
  v17 = src;
  v12 = src;
LABEL_11:
  n = 0;
  *(_BYTE *)v12 = 0;
  if ( v17 != src )
    j___libc_free_0((unsigned __int64)v17);
LABEL_3:
  if ( v21 )
    v21(v20);
  return v10;
}
