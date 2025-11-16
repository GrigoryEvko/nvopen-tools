// Function: sub_E91910
// Address: 0xe91910
//
__int64 __fastcall sub_E91910(__int64 a1, __int64 a2, __int64 a3, unsigned __int16 a4)
{
  __int64 result; // rax
  __int64 v8; // r14
  __int64 v10; // r12
  __int64 v11; // rcx
  unsigned __int16 *v12; // r11
  unsigned __int16 *v13; // rdx
  unsigned __int16 v14; // si
  __int64 v15; // rcx
  unsigned __int16 *v16; // r11

  result = a3 - 1;
  v8 = a3 & 1;
  v10 = (a3 - 1) / 2;
  if ( a2 >= v10 )
  {
    v13 = (unsigned __int16 *)(a1 + 2 * a2);
    if ( v8 )
      goto LABEL_12;
    result = a2;
    goto LABEL_15;
  }
  result = a2;
  do
  {
    v11 = 2 * (result + 1);
    v12 = (unsigned __int16 *)(a1 + 4 * (result + 1));
    result = v11 - 1;
    v13 = (unsigned __int16 *)(a1 + 2 * (v11 - 1));
    v14 = *v13;
    if ( *v12 >= *v13 )
    {
      v14 = *v12;
      v13 = v12;
      result = v11;
    }
    *(_WORD *)(a1 + v11 - 2) = v14;
  }
  while ( result < v10 );
  if ( !v8 )
  {
LABEL_15:
    if ( (a3 - 2) / 2 == result )
    {
      result = 2 * result + 1;
      *v13 = *(_WORD *)(a1 + 2 * result);
      v13 = (unsigned __int16 *)(a1 + 2 * result);
    }
  }
  v15 = (result - 1) / 2;
  if ( result > a2 )
  {
    while ( 1 )
    {
      v16 = (unsigned __int16 *)(a1 + 2 * v15);
      v13 = (unsigned __int16 *)(a1 + 2 * result);
      if ( a4 <= *v16 )
        break;
      *v13 = *v16;
      result = v15;
      if ( a2 >= v15 )
      {
        *v16 = a4;
        return result;
      }
      v15 = (v15 - 1) / 2;
    }
  }
LABEL_12:
  *v13 = a4;
  return result;
}
