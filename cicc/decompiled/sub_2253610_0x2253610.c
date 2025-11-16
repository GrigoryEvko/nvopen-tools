// Function: sub_2253610
// Address: 0x2253610
//
__int64 __fastcall sub_2253610(__int64 a1, __int64 a2, char *a3, __int64 a4, char *a5)
{
  __int64 v10; // r12
  __int64 i; // rbx
  __int64 v12; // rax
  __int64 v13; // rdx
  int v14; // eax
  int v15; // eax
  unsigned int v16; // r8d
  unsigned int v18; // eax
  const char *v19; // rdi
  const char *v20; // rsi
  int v21; // eax
  char v22; // [rsp+Ch] [rbp-3Ch]

  if ( a3 == a5 )
  {
    v19 = *(const char **)(a1 + 8);
    v20 = *(const char **)(a4 + 8);
    v16 = 6;
    if ( v19 == v20 )
      return v16;
    if ( *v19 != 42 )
    {
      v21 = strcmp(v19, v20);
      v16 = 6;
      if ( !v21 )
        return v16;
    }
  }
  v10 = *(unsigned int *)(a1 + 20);
  if ( !*(_DWORD *)(a1 + 20) )
    return 1;
  for ( i = 16 * v10 + a1; ; i -= 16 )
  {
    v12 = *(_QWORD *)(i + 16);
    if ( (v12 & 2) != 0 )
    {
      v13 = v12 >> 8;
      v14 = v12 & 1;
      v22 = v14;
      if ( !v14 )
        goto LABEL_8;
      if ( a2 != -3 )
        break;
    }
LABEL_9:
    if ( !--v10 )
      return 1;
  }
  v13 = *(_QWORD *)(*(_QWORD *)a3 + v13);
LABEL_8:
  v15 = (*(__int64 (__fastcall **)(_QWORD, __int64, char *, __int64, char *))(**(_QWORD **)(i + 8) + 64LL))(
          *(_QWORD *)(i + 8),
          a2,
          &a3[v13],
          a4,
          a5);
  v16 = v15;
  if ( v15 <= 3 )
    goto LABEL_9;
  v18 = v15 | 1;
  if ( v22 )
    return v18;
  return v16;
}
