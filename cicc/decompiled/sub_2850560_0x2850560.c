// Function: sub_2850560
// Address: 0x2850560
//
char __fastcall sub_2850560(
        __int64 *a1,
        unsigned int a2,
        __int64 a3,
        __int64 a4,
        unsigned __int64 a5,
        unsigned __int64 a6,
        char a7,
        __int64 a8,
        unsigned __int64 a9)
{
  __int64 v9; // rdi
  __int64 (*v10)(); // rax
  unsigned __int64 v11; // rcx
  char result; // al

  v11 = a6;
  if ( a2 == 2 )
  {
    if ( a7 )
      v11 = 0;
    return sub_DFA150(a1, a3, a5, v11, a8, a9);
  }
  else if ( a2 > 2 )
  {
    if ( a2 != 3 )
      BUG();
    result = 0;
    if ( !a5 )
    {
      result = a8 & (a6 != 0 && a9 != 0);
      if ( result )
      {
        return 0;
      }
      else if ( a9 + 1 <= 1 )
      {
        if ( a6 )
        {
          if ( !a7 )
          {
            if ( !a9 )
              v11 = -(__int64)a6;
            v9 = *a1;
            v10 = *(__int64 (**)())(*(_QWORD *)v9 + 448LL);
            if ( v10 == sub_DF5DD0 )
              return 0;
            else
              return ((__int64 (__fastcall *)(__int64, unsigned __int64))v10)(v9, v11);
          }
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
      return a6 == 0 && a9 + 1 <= 1;
  }
  else
  {
    return (a6 | a9 | a5) == 0;
  }
  return result;
}
