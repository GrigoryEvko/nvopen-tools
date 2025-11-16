// Function: sub_167FF60
// Address: 0x167ff60
//
__int64 __fastcall sub_167FF60(__int64 a1, char **a2, _QWORD *a3)
{
  __int64 result; // rax
  int v5; // r14d
  int v6; // ebx
  __int64 v7; // r8
  int v9; // r10d
  __int64 v10; // r9
  unsigned int i; // ecx
  __int64 v12; // r11
  int v13; // r15d
  unsigned int v14; // ecx
  char *v15; // rdi
  const void *v16; // rsi
  bool v17; // al
  size_t v18; // rdx
  int v19; // eax
  __int64 v20; // [rsp+0h] [rbp-50h]
  __int64 v21; // [rsp+8h] [rbp-48h]
  int v22; // [rsp+10h] [rbp-40h]
  unsigned int v23; // [rsp+14h] [rbp-3Ch]
  __int64 v24; // [rsp+18h] [rbp-38h]

  result = *(unsigned int *)(a1 + 24);
  if ( !(_DWORD)result )
  {
    *a3 = 0;
    return result;
  }
  v5 = *((_DWORD *)a2 + 3);
  v6 = result - 1;
  v7 = *(_QWORD *)(a1 + 8);
  v9 = 1;
  v10 = 0;
  for ( i = v5 & (result - 1); ; i = v6 & v14 )
  {
    v12 = v7 + 24LL * i;
    v13 = *(_DWORD *)(v12 + 12);
    if ( v13 != v5 )
      break;
    v15 = *a2;
    v16 = *(const void **)v12;
    v17 = *a2 + 1 == 0;
    if ( *(_QWORD *)v12 == -1 || (v17 = v15 + 2 == 0, v16 == (const void *)-2LL) )
    {
      if ( v17 )
        goto LABEL_17;
      break;
    }
    v18 = *(unsigned int *)(v12 + 8);
    if ( v18 == *((_DWORD *)a2 + 2) )
    {
      v21 = v7;
      v22 = v9;
      v23 = i;
      v24 = v10;
      if ( !*(_DWORD *)(v12 + 8)
        || (v20 = v7 + 24LL * i, v19 = memcmp(v15, v16, v18), v12 = v20, v10 = v24, i = v23, v9 = v22, v7 = v21, !v19) )
      {
LABEL_17:
        *a3 = v12;
        return 1;
      }
    }
    if ( v13 )
      goto LABEL_5;
LABEL_9:
    v14 = v9 + i;
    ++v9;
  }
  if ( v13 )
  {
LABEL_5:
    if ( v13 == 1 && *(_QWORD *)v12 == -2 && !v10 )
      v10 = v12;
    goto LABEL_9;
  }
  if ( *(_QWORD *)v12 != -1 )
    goto LABEL_9;
  if ( !v10 )
    v10 = v7 + 24LL * i;
  *a3 = v10;
  return 0;
}
