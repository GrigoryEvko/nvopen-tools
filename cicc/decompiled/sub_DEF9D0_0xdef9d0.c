// Function: sub_DEF9D0
// Address: 0xdef9d0
//
__int64 __fastcall sub_DEF9D0(_QWORD *a1)
{
  __int64 result; // rax
  char *v3; // rsi
  __int64 *v4; // rdi
  __int64 v5; // rax
  char **v6; // rdi
  char **v7; // r13
  char **v8; // rbx
  char **v9; // [rsp-58h] [rbp-58h] BYREF
  __int64 v10; // [rsp-50h] [rbp-50h]
  _BYTE v11[72]; // [rsp-48h] [rbp-48h] BYREF

  result = a1[19];
  if ( !result )
  {
    v3 = (char *)a1[15];
    v4 = (__int64 *)a1[14];
    v9 = (char **)v11;
    v10 = 0x400000000LL;
    v5 = sub_DEF990(v4, v3, (__int64)&v9);
    v6 = v9;
    a1[19] = v5;
    v7 = &v6[(unsigned int)v10];
    if ( v7 != v6 )
    {
      v8 = v6;
      do
      {
        v3 = *v8++;
        sub_DEF380((__int64)a1, (__int64)v3);
      }
      while ( v7 != v8 );
      v6 = v9;
    }
    if ( v6 != (char **)v11 )
      _libc_free(v6, v3);
    return a1[19];
  }
  return result;
}
