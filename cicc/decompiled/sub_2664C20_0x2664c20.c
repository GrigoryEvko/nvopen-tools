// Function: sub_2664C20
// Address: 0x2664c20
//
_QWORD *__fastcall sub_2664C20(char *a1, char *a2, char *a3, char *a4, _QWORD *a5)
{
  _QWORD *result; // rax
  __int64 v7; // rcx
  __int64 v8; // rcx
  __int64 v9; // r9
  __int64 v10; // rsi
  _QWORD *v11; // rcx
  __int64 v12; // r8
  __int64 v13; // r10
  __int64 v14; // rsi
  _QWORD *v15; // rcx
  __int64 v16; // rdi

  result = a5;
  if ( a4 != a3 && a2 != a1 )
  {
    do
    {
      if ( *(_QWORD *)a3 < *(_QWORD *)a1 )
      {
        *result = *(_QWORD *)a3;
        v7 = *((_QWORD *)a3 + 1);
        result += 2;
        a3 += 16;
        *(result - 1) = v7;
        if ( a2 == a1 )
          break;
      }
      else
      {
        *result = *(_QWORD *)a1;
        v8 = *((_QWORD *)a1 + 1);
        a1 += 16;
        result += 2;
        *(result - 1) = v8;
        if ( a2 == a1 )
          break;
      }
    }
    while ( a4 != a3 );
  }
  v9 = a2 - a1;
  v10 = (a2 - a1) >> 4;
  if ( v9 > 0 )
  {
    v11 = result;
    do
    {
      v12 = *(_QWORD *)a1;
      v11 += 2;
      a1 += 16;
      *(v11 - 2) = v12;
      *(v11 - 1) = *((_QWORD *)a1 - 1);
      --v10;
    }
    while ( v10 );
    result = (_QWORD *)((char *)result + v9);
  }
  v13 = a4 - a3;
  v14 = v13 >> 4;
  if ( v13 > 0 )
  {
    v15 = result;
    do
    {
      v16 = *(_QWORD *)a3;
      v15 += 2;
      a3 += 16;
      *(v15 - 2) = v16;
      *(v15 - 1) = *((_QWORD *)a3 - 1);
      --v14;
    }
    while ( v14 );
    return (_QWORD *)((char *)result + v13);
  }
  return result;
}
