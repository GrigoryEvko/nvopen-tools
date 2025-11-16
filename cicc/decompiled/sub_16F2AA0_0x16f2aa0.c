// Function: sub_16F2AA0
// Address: 0x16f2aa0
//
__int64 __fastcall sub_16F2AA0(_QWORD *a1)
{
  __int64 result; // rax
  _QWORD *v3; // rdi
  __int64 v4; // r13
  __int64 v5; // r12
  __int64 v6; // rdi

  result = *(unsigned __int8 *)a1;
  if ( (_BYTE)result == 6 )
  {
    sub_16DB620((__int64)(a1 + 1));
    return j___libc_free_0(a1[2]);
  }
  else if ( (char)result > 6 )
  {
    if ( (_BYTE)result == 7 )
    {
      v4 = a1[2];
      v5 = a1[1];
      if ( v4 != v5 )
      {
        do
        {
          v6 = v5;
          v5 += 40;
          result = sub_16F2AA0(v6);
        }
        while ( v4 != v5 );
        v5 = a1[1];
      }
      if ( v5 )
        return j_j___libc_free_0(v5, a1[3] - v5);
    }
  }
  else if ( (_BYTE)result == 5 )
  {
    v3 = (_QWORD *)a1[1];
    result = (__int64)(a1 + 3);
    if ( v3 != a1 + 3 )
      return j_j___libc_free_0(v3, a1[3] + 1LL);
  }
  return result;
}
