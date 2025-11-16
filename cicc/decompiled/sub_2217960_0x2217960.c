// Function: sub_2217960
// Address: 0x2217960
//
__int64 __fastcall sub_2217960(__int64 a1, _BYTE *a2, __int64 a3, _BYTE *a4, __int64 a5)
{
  const char *v7; // r14
  const char *v8; // rbx
  char *v9; // rbp
  char *v10; // r12
  unsigned int v11; // r15d
  const char *v12; // r14
  const char *v13; // rbx
  char *s[2]; // [rsp+10h] [rbp-78h] BYREF
  _BYTE v16[16]; // [rsp+20h] [rbp-68h] BYREF
  char *v17[2]; // [rsp+30h] [rbp-58h] BYREF
  _BYTE v18[72]; // [rsp+40h] [rbp-48h] BYREF

  s[0] = v16;
  sub_CEB5A0((__int64 *)s, a2, a3);
  v17[0] = v18;
  sub_CEB5A0((__int64 *)v17, a4, a5);
  v7 = s[0];
  v8 = v17[0];
  v9 = &s[0][(unsigned __int64)s[1]];
  v10 = &v17[0][(unsigned __int64)v17[1]];
  while ( 1 )
  {
    v11 = sub_2254F80(a1, v7, v8);
    if ( v11 )
      break;
    v12 = &v7[strlen(v7)];
    v13 = &v8[strlen(v8)];
    if ( v9 == v12 )
    {
      if ( v10 == v13 )
        break;
      if ( v9 == v12 )
      {
        v11 = -1;
        break;
      }
    }
    if ( v10 == v13 )
    {
      v11 = 1;
      break;
    }
    v7 = v12 + 1;
    v8 = v13 + 1;
  }
  if ( v17[0] != v18 )
    j___libc_free_0((unsigned __int64)v17[0]);
  if ( s[0] != v16 )
    j___libc_free_0((unsigned __int64)s[0]);
  return v11;
}
