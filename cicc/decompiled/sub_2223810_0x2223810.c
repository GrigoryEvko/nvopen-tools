// Function: sub_2223810
// Address: 0x2223810
//
__int64 __fastcall sub_2223810(__int64 a1)
{
  const wchar_t *v1; // r14
  const wchar_t *v2; // rbx
  wchar_t *v3; // rbp
  wchar_t *v4; // r12
  unsigned int v5; // r15d
  const wchar_t *v6; // r14
  const wchar_t *v7; // rbx
  wchar_t *s[2]; // [rsp+10h] [rbp-78h] BYREF
  _BYTE v10[16]; // [rsp+20h] [rbp-68h] BYREF
  wchar_t *v11[2]; // [rsp+30h] [rbp-58h] BYREF
  _BYTE v12[72]; // [rsp+40h] [rbp-48h] BYREF

  s[0] = (wchar_t *)v10;
  sub_22520E0(s);
  v11[0] = (wchar_t *)v12;
  sub_22520E0(v11);
  v1 = s[0];
  v2 = v11[0];
  v3 = &s[0][(__int64)s[1]];
  v4 = &v11[0][(__int64)v11[1]];
  while ( 1 )
  {
    v5 = sub_2254FD0(a1, v1, v2);
    if ( v5 )
      break;
    v6 = &v1[wcslen(v1)];
    v7 = &v2[wcslen(v2)];
    if ( v3 == v6 )
    {
      if ( v4 == v7 )
        break;
      if ( v3 == v6 )
      {
        v5 = -1;
        break;
      }
    }
    if ( v4 == v7 )
    {
      v5 = 1;
      break;
    }
    v1 = v6 + 1;
    v2 = v7 + 1;
  }
  if ( (_BYTE *)v11[0] != v12 )
    j___libc_free_0((unsigned __int64)v11[0]);
  if ( (_BYTE *)s[0] != v10 )
    j___libc_free_0((unsigned __int64)s[0]);
  return v5;
}
