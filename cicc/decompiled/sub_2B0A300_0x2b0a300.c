// Function: sub_2B0A300
// Address: 0x2b0a300
//
_QWORD *__fastcall sub_2B0A300(_QWORD *a1, __int64 a2, int *a3)
{
  __int64 v5; // rax
  __int64 v6; // rsi
  __int64 v7; // rax
  int v8; // eax
  _QWORD *v9; // rsi
  __int64 v10; // rcx
  int v11; // edx
  __int64 v12; // rcx
  int v13; // edx
  __int64 v14; // rcx
  int v15; // edx
  int v16; // edx
  _QWORD *result; // rax
  int v18; // ecx
  int v19; // edx
  int v20; // eax
  bool v21; // zf
  int v22; // ecx

  v5 = a2 - (_QWORD)a1;
  v6 = (a2 - (__int64)a1) >> 5;
  v7 = v5 >> 3;
  if ( v6 <= 0 )
  {
LABEL_19:
    switch ( v7 )
    {
      case 2LL:
        v19 = *a3;
        break;
      case 3LL:
        v18 = *(_DWORD *)(*a1 + 120LL);
        if ( !v18 )
          v18 = *(_DWORD *)(*a1 + 8LL);
        v19 = *a3;
        result = a1;
        if ( *a3 == v18 )
          return result;
        ++a1;
        break;
      case 1LL:
        v19 = *a3;
        goto LABEL_28;
      default:
        return (_QWORD *)a2;
    }
    v22 = *(_DWORD *)(*a1 + 120LL);
    if ( !v22 )
      v22 = *(_DWORD *)(*a1 + 8LL);
    result = a1;
    if ( v22 == v19 )
      return result;
    ++a1;
LABEL_28:
    v20 = *(_DWORD *)(*a1 + 120LL);
    if ( !v20 )
      v20 = *(_DWORD *)(*a1 + 8LL);
    v21 = v20 == v19;
    result = a1;
    if ( !v21 )
      return (_QWORD *)a2;
    return result;
  }
  v8 = *a3;
  v9 = &a1[4 * v6];
  while ( 1 )
  {
    v16 = *(_DWORD *)(*a1 + 120LL);
    if ( !v16 )
      v16 = *(_DWORD *)(*a1 + 8LL);
    if ( v8 == v16 )
      return a1;
    v10 = a1[1];
    v11 = *(_DWORD *)(v10 + 120);
    if ( !v11 )
      v11 = *(_DWORD *)(v10 + 8);
    if ( v8 == v11 )
      return a1 + 1;
    v12 = a1[2];
    v13 = *(_DWORD *)(v12 + 120);
    if ( !v13 )
      v13 = *(_DWORD *)(v12 + 8);
    if ( v8 == v13 )
      return a1 + 2;
    v14 = a1[3];
    v15 = *(_DWORD *)(v14 + 120);
    if ( !v15 )
      v15 = *(_DWORD *)(v14 + 8);
    if ( v8 == v15 )
      return a1 + 3;
    a1 += 4;
    if ( a1 == v9 )
    {
      v7 = (a2 - (__int64)a1) >> 3;
      goto LABEL_19;
    }
  }
}
