// Function: sub_37B6400
// Address: 0x37b6400
//
_DWORD *__fastcall sub_37B6400(
        __int64 a1,
        __int64 a2,
        __int64 a3,
        __int64 a4,
        __int64 a5,
        __int64 a6,
        __int64 a7,
        __int64 a8,
        int a9)
{
  __int64 v11; // r9
  __int64 i; // rdx
  __int64 v13; // rcx
  __int64 v14; // rax
  _DWORD *v15; // r8
  _DWORD *result; // rax
  int v17; // esi
  _DWORD *v18; // rdx
  __int64 v19; // rdx
  int v20; // esi
  _DWORD *v21; // rcx
  __int64 v22; // rdx
  __int64 v23; // rcx
  __int64 v24; // rdx
  _DWORD *v25; // rdx

  v11 = (a3 - 1) / 2;
  if ( a2 >= v11 )
  {
    v13 = a2;
    result = (_DWORD *)(a1 + 20 * a2);
  }
  else
  {
    for ( i = a2; ; i = v13 )
    {
      v13 = 2 * (i + 1);
      v14 = 40 * (i + 1);
      v15 = (_DWORD *)(a1 + v14 - 20);
      result = (_DWORD *)(a1 + v14);
      v17 = *result;
      if ( *result < *v15 || v17 == *v15 && result[1] < v15[1] )
      {
        --v13;
        result = (_DWORD *)(a1 + 20 * v13);
        v17 = *result;
      }
      v18 = (_DWORD *)(a1 + 20 * i);
      *v18 = v17;
      v18[1] = result[1];
      v18[2] = result[2];
      v18[3] = result[3];
      v18[4] = result[4];
      if ( v13 >= v11 )
        break;
    }
  }
  if ( (a3 & 1) == 0 && (a3 - 2) / 2 == v13 )
  {
    v22 = v13 + 1;
    v23 = 2 * (v13 + 1);
    v24 = v23 + 8 * v22;
    v13 = v23 - 1;
    v25 = (_DWORD *)(a1 + 4 * v24 - 20);
    *result = *v25;
    result[1] = v25[1];
    result[2] = v25[2];
    result[3] = v25[3];
    result[4] = v25[4];
    result = (_DWORD *)(a1 + 20 * v13);
  }
  v19 = (v13 - 1) / 2;
  if ( v13 > a2 )
  {
    while ( 1 )
    {
      result = (_DWORD *)(a1 + 20 * v19);
      v20 = *result;
      if ( *result >= (unsigned int)a7 && (v20 != (_DWORD)a7 || HIDWORD(a7) <= result[1]) )
        break;
      v21 = (_DWORD *)(a1 + 20 * v13);
      *v21 = v20;
      v21[1] = result[1];
      v21[2] = result[2];
      v21[3] = result[3];
      v21[4] = result[4];
      v13 = v19;
      if ( a2 >= v19 )
        goto LABEL_19;
      v19 = (v19 - 1) / 2;
    }
    result = (_DWORD *)(a1 + 20 * v13);
  }
LABEL_19:
  *(_QWORD *)result = a7;
  *((_QWORD *)result + 1) = a8;
  result[4] = a9;
  return result;
}
