// Function: sub_B47210
// Address: 0xb47210
//
__int64 __fastcall sub_B47210(unsigned __int8 *a1, __int64 a2, __int64 a3)
{
  __int64 result; // rax
  int v6; // r13d
  unsigned int i; // ebx
  int v8; // esi

  result = sub_B46E30((__int64)a1);
  if ( (_DWORD)result )
  {
    v6 = result;
    for ( i = 0; i != v6; ++i )
    {
      while ( 1 )
      {
        result = sub_B46EC0((__int64)a1, i);
        if ( a2 == result )
          break;
        if ( v6 == ++i )
          return result;
      }
      v8 = i;
      result = sub_B46F90(a1, v8, a3);
    }
  }
  return result;
}
