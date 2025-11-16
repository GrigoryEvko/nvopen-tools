// Function: sub_2B0E850
// Address: 0x2b0e850
//
__int64 __fastcall sub_2B0E850(__int64 a1, __int64 a2, __int64 a3, __int64 a4)
{
  __int64 result; // rax
  int v6; // r8d
  __int64 v7; // r13
  __int64 v9; // rbx
  __int64 i; // r11
  int *v11; // rdx
  int v12; // r10d
  _DWORD *v13; // rcx
  __int64 v14; // rcx
  int *v15; // r10
  __int64 v16; // rcx
  int *v17; // rcx

  result = a3 - 1;
  v6 = a4;
  v7 = a3 & 1;
  v9 = (a3 - 1) / 2;
  if ( a2 >= v9 )
  {
    v11 = (int *)(a1 + 8 * a2);
    if ( v7 )
      goto LABEL_13;
    result = a2;
    goto LABEL_16;
  }
  for ( i = a2; ; i = result )
  {
    result = 2 * (i + 1);
    v11 = (int *)(a1 + 16 * (i + 1));
    v12 = *v11;
    if ( *v11 > *(v11 - 2) )
    {
      --result;
      v11 = (int *)(a1 + 8 * result);
      v12 = *v11;
    }
    v13 = (_DWORD *)(a1 + 8 * i);
    *v13 = v12;
    v13[1] = v11[1];
    if ( result >= v9 )
      break;
  }
  if ( !v7 )
  {
LABEL_16:
    if ( (a3 - 2) / 2 == result )
    {
      v16 = result + 1;
      result = 2 * (result + 1) - 1;
      v17 = (int *)(a1 + 16 * v16 - 8);
      *v11 = *v17;
      v11[1] = v17[1];
      v11 = (int *)(a1 + 8 * result);
    }
  }
  v14 = (result - 1) / 2;
  if ( result > a2 )
  {
    while ( 1 )
    {
      v15 = (int *)(a1 + 8 * v14);
      v11 = (int *)(a1 + 8 * result);
      if ( *v15 <= v6 )
        break;
      *v11 = *v15;
      v11[1] = v15[1];
      result = v14;
      if ( a2 >= v14 )
      {
        *(_QWORD *)v15 = a4;
        return result;
      }
      v14 = (v14 - 1) / 2;
    }
  }
LABEL_13:
  *(_QWORD *)v11 = a4;
  return result;
}
