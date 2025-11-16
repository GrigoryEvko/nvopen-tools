// Function: sub_D9E120
// Address: 0xd9e120
//
__int64 __fastcall sub_D9E120(__int64 a1, __int64 a2, __int64 a3)
{
  _QWORD *v4; // rdi
  __int64 v6; // rax
  _QWORD *v7; // rdx
  __int64 v8; // rcx
  __int64 v9; // rax
  _QWORD *v10; // rcx
  _QWORD *v11; // rax
  __int64 result; // rax
  _QWORD *v13; // rcx
  __int64 v14; // rdi
  __int64 v15; // r9
  __int64 v16; // rsi
  _QWORD *v17; // rcx
  __int64 v18; // rdi

  v4 = *(_QWORD **)a1;
  v6 = 16LL * *(unsigned int *)(a1 + 8);
  v7 = &v4[(unsigned __int64)v6 / 8];
  v8 = v6 >> 4;
  v9 = v6 >> 6;
  if ( !v9 )
  {
    v11 = v4;
LABEL_9:
    if ( v8 != 2 )
    {
      if ( v8 != 3 )
      {
        if ( v8 != 1 )
          goto LABEL_12;
        goto LABEL_36;
      }
      if ( *v11 == a2 && a3 == v11[1] )
        goto LABEL_15;
      v11 += 2;
    }
    if ( *v11 == a2 && a3 == v11[1] )
      goto LABEL_15;
    v11 += 2;
LABEL_36:
    if ( *v11 == a2 && a3 == v11[1] )
      goto LABEL_15;
LABEL_12:
    v11 = v7;
LABEL_13:
    result = ((char *)v11 - (char *)v4) >> 4;
    *(_DWORD *)(a1 + 8) = result;
    return result;
  }
  v10 = &v4[8 * v9];
  v11 = v4;
  while ( *v11 != a2 || a3 != v11[1] )
  {
    if ( v11[2] == a2 && a3 == v11[3] )
    {
      v11 += 2;
      break;
    }
    if ( v11[4] == a2 && a3 == v11[5] )
    {
      v11 += 4;
      break;
    }
    if ( v11[6] == a2 && a3 == v11[7] )
    {
      v11 += 6;
      break;
    }
    v11 += 8;
    if ( v10 == v11 )
    {
      v8 = ((char *)v7 - (char *)v11) >> 4;
      goto LABEL_9;
    }
  }
LABEL_15:
  if ( v7 == v11 )
    goto LABEL_13;
  v13 = v11 + 2;
  if ( v7 == v11 + 2 )
    goto LABEL_13;
  do
  {
    while ( *v13 != a2 || a3 != v13[1] )
    {
      *v11 = *v13;
      v14 = v13[1];
      v13 += 2;
      v11 += 2;
      *(v11 - 1) = v14;
      if ( v7 == v13 )
        goto LABEL_22;
    }
    v13 += 2;
  }
  while ( v7 != v13 );
LABEL_22:
  v4 = *(_QWORD **)a1;
  v15 = *(_QWORD *)a1 + 16LL * *(unsigned int *)(a1 + 8) - (_QWORD)v7;
  v16 = v15 >> 4;
  if ( v15 <= 0 )
    goto LABEL_13;
  v17 = v11;
  do
  {
    v18 = *v7;
    v17 += 2;
    v7 += 2;
    *(v17 - 2) = v18;
    *(v17 - 1) = *(v7 - 1);
    --v16;
  }
  while ( v16 );
  result = ((__int64)v11 + v15 - *(_QWORD *)a1) >> 4;
  *(_DWORD *)(a1 + 8) = result;
  return result;
}
