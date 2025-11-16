// Function: sub_1EE73E0
// Address: 0x1ee73e0
//
char *__fastcall sub_1EE73E0(__int64 a1, unsigned __int64 a2, __int64 a3, __int64 a4, __int64 a5, int a6)
{
  int v6; // ecx
  char *result; // rax
  __int64 v11; // rdi
  char *v12; // rdx
  __int64 v13; // r8
  char *v14; // rsi
  int v15; // edx
  unsigned __int64 v16; // r14
  _QWORD *v17; // rsi
  __int64 *v18; // rdi

  v6 = a2;
  result = *(char **)a3;
  v11 = *(unsigned int *)(a3 + 8);
  v12 = (char *)(*(_QWORD *)a3 + 8 * v11);
  v13 = (8 * v11) >> 3;
  if ( (8 * v11) >> 5 )
  {
    v14 = &result[32 * ((8 * v11) >> 5)];
    while ( v6 != *(_DWORD *)result )
    {
      if ( v6 == *((_DWORD *)result + 2) )
      {
        result += 8;
        goto LABEL_8;
      }
      if ( v6 == *((_DWORD *)result + 4) )
      {
        result += 16;
        goto LABEL_8;
      }
      if ( v6 == *((_DWORD *)result + 6) )
      {
        result += 24;
        goto LABEL_8;
      }
      result += 32;
      if ( v14 == result )
      {
        v13 = (v12 - result) >> 3;
        goto LABEL_15;
      }
    }
    goto LABEL_8;
  }
LABEL_15:
  if ( v13 == 2 )
    goto LABEL_23;
  if ( v13 == 3 )
  {
    if ( (_DWORD)a2 == *(_DWORD *)result )
      goto LABEL_8;
    result += 8;
LABEL_23:
    if ( (_DWORD)a2 == *(_DWORD *)result )
      goto LABEL_8;
    result += 8;
    goto LABEL_25;
  }
  if ( v13 != 1 )
    goto LABEL_18;
LABEL_25:
  if ( (_DWORD)a2 != *(_DWORD *)result )
    goto LABEL_18;
LABEL_8:
  if ( v12 == result )
  {
LABEL_18:
    result = (char *)HIDWORD(a2);
    v16 = HIDWORD(a2);
    if ( (unsigned int)v11 >= *(_DWORD *)(a3 + 12) )
    {
      sub_16CD150(a3, (const void *)(a3 + 16), 0, 8, v13, a6);
      result = *(char **)a3;
      v12 = (char *)(*(_QWORD *)a3 + 8LL * *(unsigned int *)(a3 + 8));
    }
    *(_QWORD *)v12 = a2;
    ++*(_DWORD *)(a3 + 8);
    v17 = *(_QWORD **)(a1 + 24);
    v18 = *(__int64 **)(a1 + 48);
    goto LABEL_11;
  }
  v15 = *((_DWORD *)result + 1);
  LODWORD(v16) = v15 | HIDWORD(a2);
  *((_DWORD *)result + 1) = v15 | HIDWORD(a2);
  if ( v15 )
    return result;
  v17 = *(_QWORD **)(a1 + 24);
  v18 = *(__int64 **)(a1 + 48);
LABEL_11:
  if ( (_DWORD)v16 )
    return (char *)sub_1EE5C30(v18, v17, a2);
  return result;
}
