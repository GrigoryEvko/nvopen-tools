// Function: sub_1872550
// Address: 0x1872550
//
__int64 *__fastcall sub_1872550(__int64 a1, __int64 a2, __int64 a3, __int64 a4)
{
  __int64 v5; // r13
  __int64 v7; // rbx
  __int64 i; // r9
  __int64 v9; // rcx
  __int64 *result; // rax
  __int64 *v11; // r11
  __int64 v12; // rdx
  __int64 v13; // rdx
  __int64 *v14; // r10

  v5 = a3 & 1;
  v7 = (a3 - 1) / 2;
  if ( a2 >= v7 )
  {
    result = (__int64 *)(a1 + 8 * a2);
    if ( (a3 & 1) != 0 )
      goto LABEL_13;
    v9 = a2;
    goto LABEL_16;
  }
  for ( i = a2; ; i = v9 )
  {
    v9 = 2 * (i + 1);
    result = (__int64 *)(a1 + 16 * (i + 1));
    v11 = (__int64 *)(a1 + 8 * (v9 - 1));
    v12 = *result;
    if ( *(_DWORD *)(*result + 8) < *(_DWORD *)(*v11 + 8) )
    {
      v12 = *v11;
      --v9;
      result = v11;
    }
    *(_QWORD *)(a1 + 8 * i) = v12;
    if ( v9 >= v7 )
      break;
  }
  if ( !v5 )
  {
LABEL_16:
    if ( (a3 - 2) / 2 == v9 )
    {
      v9 = 2 * v9 + 1;
      *result = *(_QWORD *)(a1 + 8 * v9);
      result = (__int64 *)(a1 + 8 * v9);
    }
  }
  v13 = (v9 - 1) / 2;
  if ( v9 > a2 )
  {
    while ( 1 )
    {
      v14 = (__int64 *)(a1 + 8 * v13);
      result = (__int64 *)(a1 + 8 * v9);
      if ( *(_DWORD *)(*v14 + 8) >= *(_DWORD *)(a4 + 8) )
        break;
      *result = *v14;
      v9 = v13;
      if ( a2 >= v13 )
      {
        *v14 = a4;
        return (__int64 *)(a1 + 8 * v13);
      }
      v13 = (v13 - 1) / 2;
    }
  }
LABEL_13:
  *result = a4;
  return result;
}
