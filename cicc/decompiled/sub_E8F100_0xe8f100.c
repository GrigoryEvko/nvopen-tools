// Function: sub_E8F100
// Address: 0xe8f100
//
__int64 __fastcall sub_E8F100(__int64 a1, __int64 a2, __int64 a3, int *a4)
{
  __int64 v5; // r11
  __int64 i; // rcx
  __int64 v9; // rdx
  __int64 result; // rax
  __int64 v11; // rcx
  __int64 v12; // r9
  unsigned __int64 v13; // r10
  unsigned __int64 v14; // r8
  unsigned int v15; // r9d
  unsigned __int64 v16; // r8
  __int64 v17; // r10
  __int64 v18; // rcx
  unsigned __int64 v19; // rsi
  __int64 v20; // rax
  __int64 v21; // rcx

  v5 = (a3 - 1) / 2;
  if ( a2 >= v5 )
  {
    v9 = a1 + 24 * a2;
    result = a2;
  }
  else
  {
    for ( i = a2; ; i = result )
    {
      result = 2 * (i + 1) - 1;
      v12 = a1 + 48 * (i + 1);
      v9 = a1 + 24 * result;
      v13 = *(_QWORD *)(v12 + 8);
      v14 = *(_QWORD *)(v9 + 8);
      if ( v13 >= v14 )
      {
        if ( v13 == v14 )
        {
          if ( *(_DWORD *)v12 >= *(_DWORD *)v9 )
          {
            v9 = a1 + 48 * (i + 1);
            result = 2 * (i + 1);
          }
        }
        else
        {
          v14 = *(_QWORD *)(v12 + 8);
          v9 = a1 + 48 * (i + 1);
          result = 2 * (i + 1);
        }
      }
      v11 = a1 + 24 * i;
      *(_QWORD *)(v11 + 8) = v14;
      *(_DWORD *)v11 = *(_DWORD *)v9;
      *(_QWORD *)(v11 + 16) = *(_QWORD *)(v9 + 16);
      if ( result >= v5 )
        break;
    }
  }
  if ( (a3 & 1) == 0 && (a3 - 2) / 2 == result )
  {
    result = 2 * result + 1;
    v21 = a1 + 24 * result;
    *(_QWORD *)(v9 + 8) = *(_QWORD *)(v21 + 8);
    *(_DWORD *)v9 = *(_DWORD *)v21;
    *(_QWORD *)(v9 + 16) = *(_QWORD *)(v21 + 16);
    v9 = v21;
  }
  v15 = *a4;
  v16 = *((_QWORD *)a4 + 1);
  v17 = *((_QWORD *)a4 + 2);
  v18 = (result - 1) / 2;
  if ( result > a2 )
  {
    while ( 1 )
    {
      v9 = a1 + 24 * v18;
      v19 = *(_QWORD *)(v9 + 8);
      if ( v16 <= v19 && (v16 != v19 || v15 <= *(_DWORD *)v9) )
        break;
      v20 = a1 + 24 * result;
      *(_QWORD *)(v20 + 8) = v19;
      *(_DWORD *)v20 = *(_DWORD *)v9;
      *(_QWORD *)(v20 + 16) = *(_QWORD *)(v9 + 16);
      result = v18;
      if ( a2 >= v18 )
        goto LABEL_21;
      v18 = (v18 - 1) / 2;
    }
    result *= 3;
    v9 = a1 + 8 * result;
  }
LABEL_21:
  *(_QWORD *)(v9 + 8) = v16;
  *(_DWORD *)v9 = v15;
  *(_QWORD *)(v9 + 16) = v17;
  return result;
}
