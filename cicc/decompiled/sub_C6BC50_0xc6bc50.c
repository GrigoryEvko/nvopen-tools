// Function: sub_C6BC50
// Address: 0xc6bc50
//
__int64 __fastcall sub_C6BC50(unsigned __int16 *a1)
{
  __int64 result; // rax
  unsigned __int16 *v3; // rdi
  __int64 v4; // r13
  __int64 v5; // r12
  __int64 v6; // rdi

  result = *a1;
  if ( (_WORD)result == 7 )
  {
    sub_C6B900((__int64)(a1 + 4));
    return sub_C7D6A0(*((_QWORD *)a1 + 2), (unsigned __int64)*((unsigned int *)a1 + 8) << 6, 8);
  }
  else if ( (unsigned __int16)result > 7u )
  {
    if ( (_WORD)result == 8 )
    {
      v4 = *((_QWORD *)a1 + 2);
      v5 = *((_QWORD *)a1 + 1);
      if ( v4 != v5 )
      {
        do
        {
          v6 = v5;
          v5 += 40;
          result = sub_C6BC50(v6);
        }
        while ( v4 != v5 );
        v5 = *((_QWORD *)a1 + 1);
      }
      if ( v5 )
        return j_j___libc_free_0(v5, *((_QWORD *)a1 + 3) - v5);
    }
  }
  else if ( (_WORD)result == 6 )
  {
    v3 = (unsigned __int16 *)*((_QWORD *)a1 + 1);
    result = (__int64)(a1 + 12);
    if ( v3 != a1 + 12 )
      return j_j___libc_free_0(v3, *((_QWORD *)a1 + 3) + 1LL);
  }
  return result;
}
