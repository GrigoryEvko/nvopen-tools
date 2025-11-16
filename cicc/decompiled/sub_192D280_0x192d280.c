// Function: sub_192D280
// Address: 0x192d280
//
__int64 __fastcall sub_192D280(__int64 a1, __int64 a2, __int64 a3, unsigned __int64 a4)
{
  __int64 result; // rax
  __int64 v7; // r11
  __int64 i; // r10
  unsigned __int64 *v9; // rcx
  unsigned __int64 v10; // rsi
  __int64 v11; // rsi
  unsigned __int64 *v12; // r10
  __int64 v13; // rax
  unsigned __int64 v14; // rdx

  result = a3 - 1;
  v7 = (a3 - 1) / 2;
  if ( a2 >= v7 )
  {
    v9 = (unsigned __int64 *)(a1 + 8 * a2);
    if ( (a3 & 1) != 0 )
      goto LABEL_13;
    result = a2;
    goto LABEL_16;
  }
  for ( i = a2; ; i = result )
  {
    result = 2 * (i + 1);
    v9 = (unsigned __int64 *)(a1 + 16 * (i + 1));
    v10 = *v9;
    if ( *v9 < *(v9 - 1) )
    {
      --result;
      v9 = (unsigned __int64 *)(a1 + 8 * result);
      v10 = *v9;
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
      v9 = (unsigned __int64 *)(a1 + 8 * result);
    }
  }
  v11 = (result - 1) / 2;
  if ( result > a2 )
  {
    while ( 1 )
    {
      v12 = (unsigned __int64 *)(a1 + 8 * v11);
      v9 = (unsigned __int64 *)(a1 + 8 * result);
      if ( a4 <= *v12 )
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
