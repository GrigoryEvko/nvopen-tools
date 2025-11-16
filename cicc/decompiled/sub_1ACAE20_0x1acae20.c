// Function: sub_1ACAE20
// Address: 0x1acae20
//
__int64 __fastcall sub_1ACAE20(__int64 a1, __int64 a2, __int64 a3)
{
  __int64 result; // rax
  unsigned __int64 v5; // rax
  __int64 v6; // rbx

  if ( a2 == a3 )
    return 0;
  if ( !a2 )
    return 0xFFFFFFFFLL;
  result = 1;
  if ( a3 )
  {
    result = sub_1ACA9E0(a1, *(unsigned int *)(a2 + 8), *(unsigned int *)(a3 + 8));
    if ( !(_DWORD)result )
    {
      v5 = *(unsigned int *)(a2 + 8);
      if ( *(_DWORD *)(a2 + 8) )
      {
        v6 = 0;
        while ( 1 )
        {
          result = sub_1ACAA10(
                     a1,
                     *(_QWORD *)(*(_QWORD *)(a2 + 8 * (v6 - v5)) + 136LL) + 24LL,
                     *(_QWORD *)(*(_QWORD *)(a3 + 8 * (v6 - *(unsigned int *)(a3 + 8))) + 136LL) + 24LL);
          if ( (_DWORD)result )
            break;
          v5 = *(unsigned int *)(a2 + 8);
          if ( v5 <= ++v6 )
            return 0;
        }
      }
      else
      {
        return 0;
      }
    }
  }
  return result;
}
