// Function: sub_C85490
// Address: 0xc85490
//
__int64 __fastcall sub_C85490(__int64 a1, _QWORD *a2, char a3)
{
  const char **v5; // rsi
  __int64 result; // rax
  const char *v7; // rdi
  int v8; // r12d
  __int64 v9; // rbx
  _QWORD v10[4]; // [rsp+0h] [rbp-230h] BYREF
  __int16 v11; // [rsp+20h] [rbp-210h]
  _BYTE v12[32]; // [rsp+30h] [rbp-200h] BYREF
  __int16 v13; // [rsp+50h] [rbp-1E0h]
  _BYTE v14[32]; // [rsp+60h] [rbp-1D0h] BYREF
  __int16 v15; // [rsp+80h] [rbp-1B0h]
  _BYTE v16[32]; // [rsp+90h] [rbp-1A0h] BYREF
  __int16 v17; // [rsp+B0h] [rbp-180h]
  const char *v18; // [rsp+C0h] [rbp-170h] BYREF
  __int64 v19; // [rsp+C8h] [rbp-168h]
  __int64 v20; // [rsp+D0h] [rbp-160h]
  _BYTE v21[136]; // [rsp+D8h] [rbp-158h] BYREF
  const char *v22; // [rsp+160h] [rbp-D0h] BYREF
  __int64 v23; // [rsp+168h] [rbp-C8h]
  __int64 v24; // [rsp+170h] [rbp-C0h]
  char v25; // [rsp+178h] [rbp-B8h] BYREF
  __int16 v26; // [rsp+180h] [rbp-B0h]

  v18 = v21;
  v19 = 0;
  v20 = 128;
  sub_CA0EC0(a1, &v18);
  if ( a3 )
  {
    v26 = 261;
    v22 = v18;
    v23 = v19;
    if ( !(unsigned __int8)sub_C81DB0(&v22, 0) )
    {
      v23 = 0;
      v22 = &v25;
      v24 = 128;
      sub_C843A0(1, &v22);
      v17 = 257;
      v15 = 257;
      v10[0] = v18;
      v13 = 257;
      v11 = 261;
      v10[1] = v19;
      sub_C81B70(&v22, (__int64)v10, (__int64)v12, (__int64)v14, (__int64)v16);
      sub_C844F0(&v18, &v22);
      if ( v22 != &v25 )
        _libc_free(v22, &v22);
    }
  }
  v5 = &v18;
  sub_C7FCA0((__int64)a2, (__int64)&v18);
  result = a2[1];
  if ( (unsigned __int64)(result + 1) > a2[2] )
  {
    v5 = (const char **)(a2 + 3);
    sub_C8D290(a2, a2 + 3, result + 1, 1);
    result = a2[1];
  }
  *(_BYTE *)(*a2 + result) = 0;
  v7 = v18;
  if ( (_DWORD)v19 )
  {
    v8 = v19;
    v9 = 0;
    do
    {
      while ( v7[v9] != 37 )
      {
        if ( v8 == ++v9 )
          goto LABEL_9;
      }
      result = (unsigned __int8)a0123456789abcd_0[sub_C86500() & 0xF];
      *(_BYTE *)(*a2 + v9++) = result;
      v7 = v18;
    }
    while ( v8 != v9 );
  }
LABEL_9:
  if ( v7 != v21 )
    return _libc_free(v7, v5);
  return result;
}
