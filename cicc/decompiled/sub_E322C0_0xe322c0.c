// Function: sub_E322C0
// Address: 0xe322c0
//
__int64 __fastcall sub_E322C0(__int64 a1, unsigned __int64 *a2)
{
  unsigned __int64 v3; // r10
  unsigned __int64 v4; // rsi
  __int64 v5; // r9
  char v6; // al
  __int64 v7; // rdx
  __int64 result; // rax
  char v9; // bl
  __int64 v10; // rcx
  unsigned __int64 v11; // rcx
  unsigned __int64 v12; // rsi
  char v13; // bl
  __int64 v14; // rcx

  if ( *(_BYTE *)(a1 + 49) )
    goto LABEL_14;
  v3 = *(_QWORD *)(a1 + 40);
  v4 = *(_QWORD *)(a1 + 24);
  if ( v3 >= v4 )
    goto LABEL_14;
  v5 = *(_QWORD *)(a1 + 32);
  v6 = *(_BYTE *)(v5 + v3);
  if ( (unsigned __int8)(v6 - 48) > 9u )
  {
    if ( (unsigned __int8)(v6 - 97) > 5u )
    {
LABEL_14:
      *(_BYTE *)(a1 + 49) = 1;
      *a2 = 0;
      a2[1] = 0;
      return 0;
    }
    if ( v6 != 48 )
    {
LABEL_5:
      v7 = *(_QWORD *)(a1 + 40);
      result = 0;
      while ( 1 )
      {
        v9 = *(_BYTE *)(v5 + v7);
        v10 = v7++;
        *(_QWORD *)(a1 + 40) = v7;
        if ( v9 == 95 )
          break;
        v13 = *(_BYTE *)(v5 + v7 - 1);
        v14 = 16 * result;
        if ( (unsigned __int8)(v13 - 48) > 9u )
        {
          if ( (unsigned __int8)(v13 - 97) > 5u )
            goto LABEL_14;
          result = v14 + v13 - 87;
        }
        else
        {
          result = v14 + v13 - 48;
        }
        if ( v4 == v7 )
          goto LABEL_14;
      }
      v11 = v10 - v3;
      goto LABEL_8;
    }
  }
  else if ( v6 != 48 )
  {
    goto LABEL_5;
  }
  *(_QWORD *)(a1 + 40) = v3 + 1;
  if ( v4 <= v3 + 1 || *(_BYTE *)(v5 + v3 + 1) != 95 )
    goto LABEL_14;
  v11 = 1;
  *(_QWORD *)(a1 + 40) = v3 + 2;
  result = 0;
LABEL_8:
  v12 = v4 - v3;
  a2[1] = v5 + v3;
  if ( v12 > v11 )
    v12 = v11;
  *a2 = v12;
  return result;
}
