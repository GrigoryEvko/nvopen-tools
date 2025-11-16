// Function: sub_C0C4A0
// Address: 0xc0c4a0
//
__int64 __fastcall sub_C0C4A0(__int64 a1, char **a2, _QWORD *a3)
{
  __int64 result; // rax
  int v5; // r12d
  int v6; // ebx
  __int64 v7; // r8
  int v9; // r10d
  __int64 v10; // r9
  unsigned int i; // ecx
  __int64 v12; // r11
  int v13; // r15d
  char *v14; // rdi
  const void *v15; // rsi
  bool v16; // al
  size_t v17; // rdx
  unsigned int v18; // ecx
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
  for ( i = v5 & (result - 1); ; i = v6 & v18 )
  {
    v12 = v7 + 24LL * i;
    v13 = *(_DWORD *)(v12 + 12);
    if ( v13 == v5 )
    {
      v14 = *a2;
      v15 = *(const void **)v12;
      v16 = *a2 + 1 == 0;
      if ( *(_QWORD *)v12 != -1 )
      {
        v16 = v14 + 2 == 0;
        if ( v15 != (const void *)-2LL )
        {
          v17 = *((unsigned int *)a2 + 2);
          if ( (_DWORD)v17 != *(_DWORD *)(v12 + 8) )
          {
            if ( !v5 )
              goto LABEL_17;
            goto LABEL_13;
          }
          v21 = v7;
          v22 = v9;
          v23 = i;
          v24 = v10;
          if ( !*((_DWORD *)a2 + 2) )
            goto LABEL_22;
          v20 = v7 + 24LL * i;
          v19 = memcmp(v14, v15, v17);
          v7 = v21;
          v9 = v22;
          i = v23;
          v10 = v24;
          v12 = v20;
          v16 = v19 == 0;
        }
      }
      if ( v16 )
      {
LABEL_22:
        *a3 = v12;
        return 1;
      }
    }
    if ( !v13 )
      break;
LABEL_13:
    if ( v13 == 1 && *(_QWORD *)v12 == -2 && !v10 )
      v10 = v12;
LABEL_17:
    v18 = v9 + i;
    ++v9;
  }
  if ( *(_QWORD *)v12 != -1 )
    goto LABEL_17;
  if ( !v10 )
    v10 = v12;
  *a3 = v10;
  return 0;
}
