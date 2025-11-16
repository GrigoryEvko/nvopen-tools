// Function: sub_93FB40
// Address: 0x93fb40
//
char *__fastcall sub_93FB40(__int64 a1, int a2)
{
  char *v3; // rdi
  __int64 v4; // rcx
  char *v5; // rdx
  __int64 v6; // r9
  __int64 v7; // rcx
  char *result; // rax
  char *v9; // rcx
  char *v10; // rcx
  __int64 v11; // r9
  __int64 v12; // rsi
  char *v13; // rcx
  int v14; // edi

  v3 = *(char **)a1;
  v4 = 16LL * *(unsigned int *)(a1 + 8);
  v5 = &v3[v4];
  v6 = v4 >> 4;
  v7 = v4 >> 6;
  if ( v7 )
  {
    result = v3;
    v9 = &v3[64 * v7];
    while ( *(_DWORD *)result != a2 )
    {
      if ( *((_DWORD *)result + 4) == a2 )
      {
        result += 16;
        goto LABEL_8;
      }
      if ( *((_DWORD *)result + 8) == a2 )
      {
        result += 32;
        goto LABEL_8;
      }
      if ( *((_DWORD *)result + 12) == a2 )
      {
        result += 48;
        goto LABEL_8;
      }
      result += 64;
      if ( v9 == result )
      {
        v6 = (v5 - result) >> 4;
        goto LABEL_22;
      }
    }
    goto LABEL_8;
  }
  result = v3;
LABEL_22:
  if ( v6 == 2 )
  {
LABEL_29:
    if ( *(_DWORD *)result != a2 )
    {
      result += 16;
LABEL_25:
      if ( *(_DWORD *)result != a2 )
        goto LABEL_17;
      goto LABEL_8;
    }
    goto LABEL_8;
  }
  if ( v6 != 3 )
  {
    if ( v6 != 1 )
      goto LABEL_17;
    goto LABEL_25;
  }
  if ( *(_DWORD *)result != a2 )
  {
    result += 16;
    goto LABEL_29;
  }
LABEL_8:
  if ( v5 != result )
  {
    v10 = result + 16;
    if ( v5 == result + 16 )
      goto LABEL_31;
    do
    {
      if ( *(_DWORD *)v10 != a2 )
      {
        *(_DWORD *)result = *(_DWORD *)v10;
        result += 16;
        *((_QWORD *)result - 1) = *((_QWORD *)v10 + 1);
      }
      v10 += 16;
    }
    while ( v5 != v10 );
    v3 = *(char **)a1;
    v11 = *(_QWORD *)a1 + 16LL * *(unsigned int *)(a1 + 8) - (_QWORD)v5;
    v12 = v11 >> 4;
    if ( v11 <= 0 )
    {
LABEL_31:
      v5 = result;
    }
    else
    {
      v13 = result;
      do
      {
        v14 = *(_DWORD *)v5;
        v13 += 16;
        v5 += 16;
        *((_DWORD *)v13 - 4) = v14;
        *((_QWORD *)v13 - 1) = *((_QWORD *)v5 - 1);
        --v12;
      }
      while ( v12 );
      v3 = *(char **)a1;
      v5 = &result[v11];
    }
  }
LABEL_17:
  *(_DWORD *)(a1 + 8) = (v5 - v3) >> 4;
  return result;
}
