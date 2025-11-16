// Function: sub_B463C0
// Address: 0xb463c0
//
__int64 __fastcall sub_B463C0(__int64 a1, __int64 a2)
{
  __int64 result; // rax
  __int64 v3; // rdx

  result = *(_QWORD *)(a1 + 16);
  if ( result )
  {
    while ( 1 )
    {
      v3 = *(_QWORD *)(result + 24);
      if ( *(_BYTE *)v3 == 84 )
      {
        if ( a2 != *(_QWORD *)(*(_QWORD *)(v3 - 8)
                             + 32LL * *(unsigned int *)(v3 + 72)
                             + 8LL * (unsigned int)((result - *(_QWORD *)(v3 - 8)) >> 5)) )
          return 1;
      }
      else if ( a2 != *(_QWORD *)(v3 + 40) )
      {
        return 1;
      }
      result = *(_QWORD *)(result + 8);
      if ( !result )
        return result;
    }
  }
  return 0;
}
