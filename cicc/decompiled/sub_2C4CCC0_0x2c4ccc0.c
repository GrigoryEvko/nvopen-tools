// Function: sub_2C4CCC0
// Address: 0x2c4ccc0
//
__int64 __fastcall sub_2C4CCC0(__int64 a1, __int64 a2, __int64 a3, unsigned int a4)
{
  __int64 result; // rax
  __int64 v7; // r11
  __int64 i; // r10
  unsigned int *v9; // rcx
  unsigned int v10; // esi
  __int64 v11; // rsi
  unsigned int *v12; // r10
  __int64 v13; // rax
  unsigned int v14; // edx

  result = a3 - 1;
  v7 = (a3 - 1) / 2;
  if ( a2 >= v7 )
  {
    v9 = (unsigned int *)(a1 + 4 * a2);
    if ( (a3 & 1) != 0 )
      goto LABEL_13;
    result = a2;
    goto LABEL_16;
  }
  for ( i = a2; ; i = result )
  {
    result = 2 * (i + 1);
    v9 = (unsigned int *)(a1 + 8 * (i + 1));
    v10 = *v9;
    if ( *v9 < *(v9 - 1) )
    {
      --result;
      v9 = (unsigned int *)(a1 + 4 * result);
      v10 = *v9;
    }
    *(_DWORD *)(a1 + 4 * i) = v10;
    if ( result >= v7 )
      break;
  }
  if ( (a3 & 1) == 0 )
  {
LABEL_16:
    if ( result == (a3 - 2) / 2 )
    {
      v13 = 2 * result + 2;
      v14 = *(_DWORD *)(a1 + 4 * v13 - 4);
      result = v13 - 1;
      *v9 = v14;
      v9 = (unsigned int *)(a1 + 4 * result);
    }
  }
  v11 = (result - 1) / 2;
  if ( result > a2 )
  {
    while ( 1 )
    {
      v12 = (unsigned int *)(a1 + 4 * v11);
      v9 = (unsigned int *)(a1 + 4 * result);
      if ( *v12 >= a4 )
        break;
      *v9 = *v12;
      result = v11;
      if ( a2 >= v11 )
      {
        *v12 = a4;
        return result;
      }
      v11 = (v11 - 1) / 2;
    }
  }
LABEL_13:
  *v9 = a4;
  return result;
}
