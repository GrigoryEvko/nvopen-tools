// Function: sub_18903D0
// Address: 0x18903d0
//
char *__fastcall sub_18903D0(char *a1, char *a2, char *a3, char *a4, _QWORD *a5)
{
  __int64 v5; // rax
  __int64 v6; // rax
  __int64 v7; // r9
  __int64 v8; // rsi
  _QWORD *v9; // rcx
  __int64 v10; // rax
  __int64 v12; // r10
  __int64 v13; // rcx
  _QWORD *v14; // rdx
  __int64 v15; // r9

  if ( a1 == a2 )
  {
LABEL_7:
    v7 = a4 - a3;
    v8 = (a4 - a3) >> 4;
    if ( a4 - a3 > 0 )
    {
      v9 = a5;
      do
      {
        v10 = *(_QWORD *)a3;
        v9 += 2;
        a3 += 16;
        *(v9 - 2) = v10;
        *(v9 - 1) = *((_QWORD *)a3 - 1);
        --v8;
      }
      while ( v8 );
      if ( v7 <= 0 )
        v7 = 16;
      return (char *)a5 + v7;
    }
    return (char *)a5;
  }
  while ( a4 != a3 )
  {
    if ( *(_QWORD *)a3 < *(_QWORD *)a1 )
    {
      *a5 = *(_QWORD *)a3;
      v5 = *((_QWORD *)a3 + 1);
      a5 += 2;
      a3 += 16;
      *(a5 - 1) = v5;
      if ( a1 == a2 )
        goto LABEL_7;
    }
    else
    {
      *a5 = *(_QWORD *)a1;
      v6 = *((_QWORD *)a1 + 1);
      a1 += 16;
      a5 += 2;
      *(a5 - 1) = v6;
      if ( a1 == a2 )
        goto LABEL_7;
    }
  }
  v12 = a2 - a1;
  v13 = (a2 - a1) >> 4;
  if ( a2 - a1 <= 0 )
    return (char *)a5;
  v14 = a5;
  do
  {
    v15 = *(_QWORD *)a1;
    v14 += 2;
    a1 += 16;
    *(v14 - 2) = v15;
    *(v14 - 1) = *((_QWORD *)a1 - 1);
    --v13;
  }
  while ( v13 );
  if ( v12 <= 0 )
    v12 = 16;
  return (char *)a5 + v12;
}
