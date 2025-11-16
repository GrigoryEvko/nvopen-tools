// Function: sub_985A60
// Address: 0x985a60
//
char *__fastcall sub_985A60(char **a1, __int64 a2, char **a3, unsigned __int64 *a4, char **a5)
{
  char *result; // rax
  __int64 v7; // rcx
  unsigned __int64 v8; // rax
  int v9; // ecx
  unsigned __int64 v10; // rax
  char v11; // cl
  char *v12; // rdi
  char *v13; // rcx
  char **v14; // rcx
  char *v15; // rdi
  char *v16; // rcx
  __int64 v17; // rcx
  __int64 v18; // rdx
  unsigned __int64 v19; // rax
  int v20; // edx

  result = *a1;
  *a3 = *a1;
  if ( (char *)a2 == result )
    return result;
  v7 = *(_QWORD *)(*(_QWORD *)(a2 - 8)
                 + 32LL * *(unsigned int *)(a2 + 72)
                 + 8LL * (unsigned int)(((__int64)a1 - *(_QWORD *)(a2 - 8)) >> 5));
  v8 = *(_QWORD *)(v7 + 48) & 0xFFFFFFFFFFFFFFF8LL;
  if ( v8 == v7 + 48 )
  {
    v10 = 0;
  }
  else
  {
    if ( !v8 )
      goto LABEL_40;
    v9 = *(unsigned __int8 *)(v8 - 24);
    v10 = v8 - 24;
    if ( (unsigned int)(v9 - 30) >= 0xB )
      v10 = 0;
  }
  *a4 = v10;
  if ( a5 )
    *a5 = (char *)a2;
  result = *a3;
  v11 = **a3;
  if ( v11 == 86 )
  {
    if ( (result[7] & 0x40) != 0 )
    {
      v12 = (char *)*((_QWORD *)result - 1);
      v13 = (char *)*((_QWORD *)v12 + 4);
      if ( (char *)a2 == v13 )
      {
        result = (char *)*((_QWORD *)v12 + 8);
        if ( result )
          goto LABEL_19;
      }
      if ( !v13 )
        return result;
    }
    else
    {
      result -= 32 * (*((_DWORD *)result + 1) & 0x7FFFFFF);
      v13 = (char *)*((_QWORD *)result + 4);
      v12 = result;
      if ( (char *)a2 == v13 )
      {
        result = (char *)*((_QWORD *)result + 8);
        if ( result )
          goto LABEL_19;
      }
      if ( !v13 )
        return result;
    }
    if ( a2 != *((_QWORD *)v12 + 8) )
      return result;
    result = v13;
LABEL_19:
    *a3 = result;
    if ( *result != 84 )
      return result;
    goto LABEL_20;
  }
  if ( v11 != 84 )
    return result;
LABEL_20:
  if ( (*((_DWORD *)result + 1) & 0x7FFFFFF) == 2 )
  {
    v14 = (char **)*((_QWORD *)result - 1);
    v15 = *v14;
    if ( *v14 && (char *)a2 == v15 )
    {
      v15 = v14[4];
      v17 = 8;
      goto LABEL_26;
    }
    v16 = v14[4];
    if ( (char *)a2 == v16 && v16 )
    {
      v17 = 0;
LABEL_26:
      *a3 = v15;
      if ( a5 )
        *a5 = result;
      v18 = *(_QWORD *)(*((_QWORD *)result - 1) + 32LL * *((unsigned int *)result + 18) + v17);
      v19 = *(_QWORD *)(v18 + 48) & 0xFFFFFFFFFFFFFFF8LL;
      if ( v19 == v18 + 48 )
      {
        result = 0;
LABEL_32:
        *a4 = (unsigned __int64)result;
        return result;
      }
      if ( v19 )
      {
        v20 = *(unsigned __int8 *)(v19 - 24);
        result = (char *)(v19 - 24);
        if ( (unsigned int)(v20 - 30) >= 0xB )
          result = 0;
        goto LABEL_32;
      }
LABEL_40:
      BUG();
    }
  }
  return result;
}
