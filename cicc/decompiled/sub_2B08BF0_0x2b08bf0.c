// Function: sub_2B08BF0
// Address: 0x2b08bf0
//
_DWORD *__fastcall sub_2B08BF0(_DWORD *a1, __int64 a2, __int64 a3)
{
  __int64 v4; // rdx
  __int64 v5; // rax
  _DWORD *v6; // rdx
  int v7; // eax
  _DWORD *result; // rax
  int v9; // edx

  v4 = (a2 - (__int64)a1) >> 6;
  v5 = (a2 - (__int64)a1) >> 4;
  if ( v4 <= 0 )
  {
LABEL_11:
    switch ( v5 )
    {
      case 2LL:
        v9 = *(_DWORD *)(*(_QWORD *)(a3 + 272) + 8LL);
        result = a1;
        break;
      case 3LL:
        v9 = *(_DWORD *)(*(_QWORD *)(a3 + 272) + 8LL);
        result = a1;
        if ( a1[2] != v9 )
          return result;
        result = a1 + 4;
        break;
      case 1LL:
        v9 = *(_DWORD *)(*(_QWORD *)(a3 + 272) + 8LL);
LABEL_18:
        result = a1;
        if ( a1[2] == v9 )
          return (_DWORD *)a2;
        return result;
      default:
        return (_DWORD *)a2;
    }
    if ( result[2] != v9 )
      return result;
    a1 = result + 4;
    goto LABEL_18;
  }
  v6 = &a1[16 * v4];
  v7 = *(_DWORD *)(*(_QWORD *)(a3 + 272) + 8LL);
  while ( 1 )
  {
    if ( a1[2] != v7 )
      return a1;
    if ( v7 != a1[6] )
      return a1 + 4;
    if ( v7 != a1[10] )
      return a1 + 8;
    if ( v7 != a1[14] )
      return a1 + 12;
    a1 += 16;
    if ( a1 == v6 )
    {
      v5 = (a2 - (__int64)a1) >> 4;
      goto LABEL_11;
    }
  }
}
