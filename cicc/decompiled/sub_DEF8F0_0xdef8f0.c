// Function: sub_DEF8F0
// Address: 0xdef8f0
//
__int64 __fastcall sub_DEF8F0(_QWORD *a1)
{
  __int64 result; // rax
  __int64 v3; // rsi
  __int64 *v4; // rdi
  __int64 v5; // rax
  __int64 *v6; // rdi
  __int64 *v7; // r13
  __int64 *v8; // rbx
  __int64 *v9; // [rsp-58h] [rbp-58h] BYREF
  __int64 v10; // [rsp-50h] [rbp-50h]
  _BYTE v11[72]; // [rsp-48h] [rbp-48h] BYREF

  result = a1[18];
  if ( !result )
  {
    v3 = a1[15];
    v4 = (__int64 *)a1[14];
    v9 = (__int64 *)v11;
    v10 = 0x400000000LL;
    v5 = sub_DEF8B0(v4, v3, (__int64)&v9);
    v6 = v9;
    a1[18] = v5;
    v7 = &v6[(unsigned int)v10];
    if ( v7 != v6 )
    {
      v8 = v6;
      do
      {
        v3 = *v8++;
        sub_DEF380((__int64)a1, v3);
      }
      while ( v7 != v8 );
      v6 = v9;
    }
    if ( v6 != (__int64 *)v11 )
      _libc_free(v6, v3);
    return a1[18];
  }
  return result;
}
