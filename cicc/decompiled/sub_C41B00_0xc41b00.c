// Function: sub_C41B00
// Address: 0xc41b00
//
double __fastcall sub_C41B00(__int64 *a1)
{
  _DWORD *v1; // rbx
  _DWORD *v2; // r12
  __int64 *v3; // rdi
  double v4; // r13
  void **v6; // rbx
  bool v7; // [rsp+Fh] [rbp-41h] BYREF
  _DWORD *v8; // [rsp+10h] [rbp-40h] BYREF
  __int64 *v9; // [rsp+18h] [rbp-38h]

  v1 = (_DWORD *)*a1;
  v2 = sub_C33340();
  if ( v1 == dword_3F657A0 )
  {
    if ( v2 == v1 )
      a1 = (__int64 *)a1[1];
    return sub_C3BA60(a1);
  }
  else
  {
    if ( v1 == v2 )
      sub_C3C790(&v8, (_QWORD **)a1);
    else
      sub_C33EB0(&v8, a1);
    sub_C41640((__int64 *)&v8, dword_3F657A0, 1, &v7);
    v3 = (__int64 *)&v8;
    if ( v8 == v2 )
      v3 = v9;
    v4 = sub_C3BA60(v3);
    if ( v8 == v2 )
    {
      if ( v9 )
      {
        v6 = (void **)&v9[3 * *(v9 - 1)];
        while ( v9 != (__int64 *)v6 )
        {
          v6 -= 3;
          if ( v2 == *v6 )
            sub_969EE0((__int64)v6);
          else
            sub_C338F0((__int64)v6);
        }
        j_j_j___libc_free_0_0(v6 - 1);
      }
    }
    else
    {
      sub_C338F0((__int64)&v8);
    }
    return v4;
  }
}
