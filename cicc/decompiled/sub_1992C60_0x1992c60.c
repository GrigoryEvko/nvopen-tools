// Function: sub_1992C60
// Address: 0x1992c60
//
char __fastcall sub_1992C60(
        __int64 *a1,
        unsigned int a2,
        __int64 a3,
        __int64 a4,
        unsigned __int64 a5,
        unsigned __int64 a6,
        unsigned int a7,
        unsigned __int64 a8)
{
  __int64 v8; // rdi
  __int64 (*v9)(); // rax
  unsigned __int64 v10; // r10
  char result; // al

  v10 = a6;
  if ( a2 == 2 )
    return sub_14A2A90(a1, a3, a5, a6, a7, a8);
  if ( a2 > 2 )
  {
    result = 0;
    if ( !a5 )
    {
      result = a7 & (a8 != 0 && a6 != 0);
      if ( result )
      {
        return 0;
      }
      else if ( a8 + 1 <= 1 )
      {
        if ( a6 )
        {
          if ( !a8 )
            v10 = -(__int64)a6;
          v8 = *a1;
          v9 = *(__int64 (**)())(*(_QWORD *)v8 + 168LL);
          if ( v9 == sub_14A0840 )
            return 0;
          else
            return ((__int64 (__fastcall *)(__int64, unsigned __int64, __int64 (*)(), __int64, _QWORD, unsigned __int64))v9)(
                     v8,
                     v10,
                     sub_14A0840,
                     a4,
                     a7,
                     a8);
        }
        else
        {
          return 1;
        }
      }
    }
  }
  else if ( a2 )
  {
    result = 0;
    if ( !a5 )
      return a6 == 0 && a8 + 1 <= 1;
  }
  else
  {
    return (a6 | a8 | a5) == 0;
  }
  return result;
}
