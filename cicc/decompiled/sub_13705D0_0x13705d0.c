// Function: sub_13705D0
// Address: 0x13705d0
//
__int64 __fastcall sub_13705D0(__int64 a1, __int64 a2, __int64 a3, unsigned int a4)
{
  __int64 result; // rax
  __int64 v5; // r8
  __int64 v7; // r9
  unsigned int *v8; // rcx
  __int64 v9; // rsi
  unsigned int *v10; // r9
  __int64 v11; // rax
  unsigned int v12; // edx

  result = a3 - 1;
  v5 = a2;
  v7 = (a3 - 1) / 2;
  if ( a2 >= v7 )
  {
    v8 = (unsigned int *)(a1 + 4 * a2);
    if ( (a3 & 1) != 0 )
      goto LABEL_13;
    result = a2;
    goto LABEL_16;
  }
  while ( 1 )
  {
    result = 2 * (a2 + 1);
    v8 = (unsigned int *)(a1 + 8 * (a2 + 1));
    if ( *v8 < *(v8 - 1) )
    {
      --result;
      v8 = (unsigned int *)(a1 + 4 * result);
    }
    *(_DWORD *)(a1 + 4 * a2) = *v8;
    if ( result >= v7 )
      break;
    a2 = result;
  }
  if ( (a3 & 1) == 0 )
  {
LABEL_16:
    if ( (a3 - 2) / 2 == result )
    {
      v11 = 2 * result + 2;
      v12 = *(_DWORD *)(a1 + 4 * v11 - 4);
      result = v11 - 1;
      *v8 = v12;
      v8 = (unsigned int *)(a1 + 4 * result);
    }
  }
  v9 = (result - 1) / 2;
  if ( result > v5 )
  {
    while ( 1 )
    {
      v10 = (unsigned int *)(a1 + 4 * v9);
      v8 = (unsigned int *)(a1 + 4 * result);
      if ( *v10 >= a4 )
        break;
      *v8 = *v10;
      result = v9;
      if ( v5 >= v9 )
      {
        *v10 = a4;
        return result;
      }
      v9 = (v9 - 1) / 2;
    }
  }
LABEL_13:
  *v8 = a4;
  return result;
}
