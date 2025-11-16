// Function: sub_2FE5F90
// Address: 0x2fe5f90
//
__int64 __fastcall sub_2FE5F90(__int64 a1, unsigned int a2, __int64 a3)
{
  __int64 result; // rax
  __int64 v4; // rdx
  bool v5; // cc
  __int64 v6; // rax

  switch ( a3 )
  {
    case 1LL:
      v4 = 0;
      goto LABEL_5;
    case 2LL:
      v4 = 1;
      goto LABEL_5;
    case 4LL:
      v4 = 2;
      v5 = a2 <= 5;
      if ( a2 == 5 )
        return *(unsigned int *)(a1 + 4 * (4 * v4 + 2));
      goto LABEL_6;
    case 8LL:
      v4 = 3;
      goto LABEL_5;
    case 16LL:
      v4 = 4;
LABEL_5:
      v5 = a2 <= 5;
      if ( a2 == 5 )
        return *(unsigned int *)(a1 + 4 * (4 * v4 + 2));
LABEL_6:
      if ( v5 )
      {
        if ( a2 == 2 )
        {
          v6 = 0;
        }
        else
        {
          if ( a2 != 4 )
            return 729;
          v6 = 1;
        }
        return *(unsigned int *)(a1 + 4 * (v6 + 4 * v4));
      }
      else
      {
        result = 729;
        if ( a2 - 6 <= 1 )
          return *(unsigned int *)(a1 + 4 * (4 * v4 + 3));
      }
      return result;
    default:
      return 729;
  }
}
