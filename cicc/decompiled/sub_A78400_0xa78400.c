// Function: sub_A78400
// Address: 0xa78400
//
unsigned __int64 __fastcall sub_A78400(_QWORD *a1, int a2, unsigned __int64 a3)
{
  unsigned __int64 *v3; // rcx
  unsigned __int64 v5; // rdx
  unsigned __int64 *v6; // rax
  unsigned __int64 *v7; // rdx
  unsigned __int64 *v8; // rsi
  unsigned __int64 result; // rax
  unsigned __int64 v10; // [rsp-90h] [rbp-90h]
  unsigned __int64 *v11; // [rsp-88h] [rbp-88h] BYREF
  __int64 v12; // [rsp-80h] [rbp-80h]
  _BYTE v13[120]; // [rsp-78h] [rbp-78h] BYREF

  if ( !a3 )
    return 0;
  v3 = (unsigned __int64 *)v13;
  v5 = (unsigned int)(a2 + 2);
  v11 = (unsigned __int64 *)v13;
  v12 = 0x800000000LL;
  if ( a2 != -2 )
  {
    v6 = (unsigned __int64 *)v13;
    if ( v5 > 8 )
    {
      sub_C8D5F0(&v11, v13, v5, 8);
      v3 = v11;
      v5 = (unsigned int)(a2 + 2);
      v6 = &v11[(unsigned int)v12];
    }
    v7 = &v3[v5];
    if ( v7 != v6 )
    {
      do
      {
        if ( v6 )
          *v6 = 0;
        ++v6;
      }
      while ( v7 != v6 );
      v3 = v11;
    }
    LODWORD(v12) = a2 + 2;
  }
  v3[a2 + 1] = a3;
  v8 = v11;
  result = sub_A77EC0(a1, v11, (unsigned int)v12);
  if ( v11 != (unsigned __int64 *)v13 )
  {
    v10 = result;
    _libc_free(v11, v8);
    return v10;
  }
  return result;
}
