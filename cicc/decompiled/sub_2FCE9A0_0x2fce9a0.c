// Function: sub_2FCE9A0
// Address: 0x2fce9a0
//
__int64 __fastcall sub_2FCE9A0(__int64 a1, __int64 a2, __int64 a3, unsigned int *a4)
{
  __int64 result; // rax
  __int64 v7; // r11
  __int64 i; // r10
  _QWORD *v9; // rcx
  _DWORD *v10; // rsi
  __int64 v11; // rsi
  _DWORD **v12; // r10
  __int64 v13; // rax
  __int64 v14; // rdx

  result = a3 - 1;
  v7 = (a3 - 1) / 2;
  if ( a2 >= v7 )
  {
    v9 = (_QWORD *)(a1 + 8 * a2);
    if ( (a3 & 1) != 0 )
      goto LABEL_13;
    result = a2;
    goto LABEL_16;
  }
  for ( i = a2; ; i = result )
  {
    result = 2 * (i + 1);
    v9 = (_QWORD *)(a1 + 16 * (i + 1));
    v10 = (_DWORD *)*v9;
    if ( *(_DWORD *)*v9 < *(_DWORD *)*(v9 - 1) )
    {
      --result;
      v9 = (_QWORD *)(a1 + 8 * result);
      v10 = (_DWORD *)*v9;
    }
    *(_QWORD *)(a1 + 8 * i) = v10;
    if ( result >= v7 )
      break;
  }
  if ( (a3 & 1) == 0 )
  {
LABEL_16:
    if ( (a3 - 2) / 2 == result )
    {
      v13 = 2 * result + 2;
      v14 = *(_QWORD *)(a1 + 8 * v13 - 8);
      result = v13 - 1;
      *v9 = v14;
      v9 = (_QWORD *)(a1 + 8 * result);
    }
  }
  v11 = (result - 1) / 2;
  if ( result > a2 )
  {
    while ( 1 )
    {
      v12 = (_DWORD **)(a1 + 8 * v11);
      v9 = (_QWORD *)(a1 + 8 * result);
      result = *a4;
      if ( **v12 >= (int)result )
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
