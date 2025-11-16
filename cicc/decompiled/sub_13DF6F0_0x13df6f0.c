// Function: sub_13DF6F0
// Address: 0x13df6f0
//
_QWORD *__fastcall sub_13DF6F0(int a1, unsigned __int8 *a2, unsigned __int8 *a3, _QWORD *a4, int a5)
{
  __int64 v6; // rdx
  unsigned __int8 *v7; // r13
  __int64 v8; // rax
  unsigned __int8 **v9; // rdx
  unsigned int v10; // r14d
  unsigned __int8 **v11; // rbx
  _QWORD *v12; // r15
  _QWORD *v13; // rax
  unsigned __int8 **v17; // [rsp+18h] [rbp-38h]

  if ( !a5 )
    return 0;
  v6 = a4[2];
  if ( a2[16] == 77 )
  {
    v7 = a2;
    if ( !sub_13CB700((__int64)a3, (__int64)a2, v6) )
      return 0;
  }
  else
  {
    v7 = a3;
    if ( !sub_13CB700((__int64)a2, (__int64)a3, v6) )
      return 0;
  }
  v8 = 3LL * (*((_DWORD *)v7 + 5) & 0xFFFFFFF);
  if ( (v7[23] & 0x40) != 0 )
  {
    v9 = (unsigned __int8 **)*((_QWORD *)v7 - 1);
    v17 = &v9[v8];
  }
  else
  {
    v17 = (unsigned __int8 **)v7;
    v9 = (unsigned __int8 **)&v7[-(v8 * 8)];
  }
  if ( v17 != v9 )
  {
    v10 = a5 - 1;
    v11 = v9;
    v12 = 0;
    while ( 1 )
    {
      if ( v7 != *v11 )
      {
        if ( v7 == a2 )
        {
          v13 = sub_13DDBD0(a1, *v11, a3, a4, v10);
          if ( !v13 )
            return 0;
        }
        else
        {
          v13 = sub_13DDBD0(a1, a2, *v11, a4, v10);
          if ( !v13 )
            return 0;
        }
        if ( v12 && v13 != v12 )
          return 0;
        v12 = v13;
      }
      v11 += 3;
      if ( v17 == v11 )
        return v12;
    }
  }
  return 0;
}
