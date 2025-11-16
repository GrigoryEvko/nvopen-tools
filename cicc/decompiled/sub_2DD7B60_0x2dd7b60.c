// Function: sub_2DD7B60
// Address: 0x2dd7b60
//
__int64 __fastcall sub_2DD7B60(__int64 a1, int *a2, _QWORD *a3)
{
  int v4; // ebx
  __int64 v6; // r12
  unsigned int v8; // eax
  int v9; // r15d
  int v10; // ecx
  int *v11; // r9
  int v12; // r10d
  unsigned __int64 v13; // rax
  unsigned int i; // ebx
  int *v15; // r8
  int v16; // r11d
  const void *v17; // rsi
  const void *v18; // rdi
  size_t v19; // rdx
  unsigned int v20; // ebx
  int v21; // eax
  int v22; // [rsp+Ch] [rbp-44h]
  int *v23; // [rsp+10h] [rbp-40h]
  int v24; // [rsp+18h] [rbp-38h]
  int v25; // [rsp+1Ch] [rbp-34h]

  v4 = *(_DWORD *)(a1 + 24);
  if ( !v4 )
  {
    *a3 = 0;
    return 0;
  }
  v6 = *(_QWORD *)(a1 + 8);
  v8 = sub_C94890(*((_QWORD **)a2 + 1), *((_QWORD *)a2 + 2));
  v9 = *a2;
  v10 = v4 - 1;
  v11 = 0;
  v12 = 1;
  v13 = 0xBF58476D1CE4E5B9LL * (v8 | ((unsigned __int64)(unsigned int)(37 * *a2) << 32));
  for ( i = (v4 - 1) & ((v13 >> 31) ^ v13); ; i = v10 & v20 )
  {
    v15 = (int *)(v6 + 32LL * i);
    v16 = *v15;
    if ( *v15 == v9 )
    {
      v17 = (const void *)*((_QWORD *)v15 + 1);
      v18 = (const void *)*((_QWORD *)a2 + 1);
      if ( v17 == (const void *)-1LL )
      {
        if ( v18 == (const void *)-1LL )
          goto LABEL_17;
      }
      else if ( v17 == (const void *)-2LL )
      {
        if ( v18 == (const void *)-2LL )
          goto LABEL_17;
      }
      else
      {
        v19 = *((_QWORD *)a2 + 2);
        if ( *((_QWORD *)v15 + 2) != v19 )
        {
          if ( v9 == -1 )
            goto LABEL_15;
          goto LABEL_6;
        }
        v22 = *v15;
        v24 = v12;
        v23 = v11;
        v25 = v10;
        if ( !v19
          || (v21 = memcmp(v18, v17, v19), v15 = (int *)(v6 + 32LL * i),
                                           v10 = v25,
                                           v11 = v23,
                                           v12 = v24,
                                           v16 = v22,
                                           !v21) )
        {
LABEL_17:
          *a3 = v15;
          return 1;
        }
      }
    }
    if ( v16 == -1 )
      break;
LABEL_6:
    if ( v16 == -2 && *((_QWORD *)v15 + 1) == -2 && !v11 )
      v11 = v15;
LABEL_15:
    v20 = v12 + i;
    ++v12;
  }
  if ( *((_QWORD *)v15 + 1) != -1 )
    goto LABEL_15;
  if ( !v11 )
    v11 = v15;
  *a3 = v11;
  return 0;
}
