// Function: sub_22AF1D0
// Address: 0x22af1d0
//
__int64 __fastcall sub_22AF1D0(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5, unsigned __int64 a6)
{
  unsigned __int8 *v7; // r15
  __int64 result; // rax
  int v9; // eax
  __int64 v10; // rdx
  unsigned __int8 *v11; // r12
  unsigned __int8 *v12; // r13
  __int64 v13; // r15
  unsigned __int64 v14; // rcx
  __int64 v15; // rdx
  char *v16; // rcx
  char *v17; // rax
  char *v18; // rdx
  char *v19; // rsi
  __int64 v20; // r9
  __int64 v21; // rdi
  __int64 v22; // r12
  __int64 v23; // r13
  __int64 *v24; // r12
  __int64 *v25; // r13
  __int64 v26; // r14
  __int64 v27; // rdx
  char *v28; // [rsp+0h] [rbp-40h]
  const void *v29; // [rsp+8h] [rbp-38h]

  v7 = *(unsigned __int8 **)(a1 + 16);
  result = *v7;
  if ( (unsigned __int8)(result - 82) <= 1u )
  {
    v9 = sub_22AF1A0(*(_QWORD *)(a1 + 16));
    if ( v9 != (*((_WORD *)v7 + 1) & 0x3F) )
    {
      *(_DWORD *)(a1 + 76) = v9;
      *(_BYTE *)(a1 + 80) = 1;
    }
    v7 = *(unsigned __int8 **)(a1 + 16);
    result = *v7;
  }
  v10 = 32LL * (*((_DWORD *)v7 + 1) & 0x7FFFFFF);
  if ( (v7[7] & 0x40) != 0 )
  {
    v11 = (unsigned __int8 *)*((_QWORD *)v7 - 1);
    v12 = &v11[v10];
  }
  else
  {
    v12 = v7;
    v11 = &v7[-v10];
  }
  if ( v11 != v12 )
  {
    v29 = (const void *)(a1 + 40);
    do
    {
      v27 = *(unsigned int *)(a1 + 32);
      v13 = *(_QWORD *)v11;
      v14 = *(unsigned int *)(a1 + 36);
      a6 = v27 + 1;
      if ( (unsigned __int8)(result - 82) > 1u || !*(_BYTE *)(a1 + 80) )
      {
        if ( a6 > v14 )
        {
          sub_C8D5F0(a1 + 24, v29, v27 + 1, 8u, a5, a6);
          v27 = *(unsigned int *)(a1 + 32);
        }
        *(_QWORD *)(*(_QWORD *)(a1 + 24) + 8 * v27) = v13;
        ++*(_DWORD *)(a1 + 32);
        goto LABEL_12;
      }
      v15 = 8 * v27;
      if ( !v15 )
      {
        if ( a6 > v14 )
        {
          sub_C8D5F0(a1 + 24, v29, a6, 8u, a5, a6);
          v15 = 8LL * *(unsigned int *)(a1 + 32);
        }
        *(_QWORD *)(*(_QWORD *)(a1 + 24) + v15) = v13;
        ++*(_DWORD *)(a1 + 32);
        goto LABEL_12;
      }
      if ( a6 > v14 )
      {
        sub_C8D5F0(a1 + 24, v29, a6, 8u, a5, a6);
        v19 = *(char **)(a1 + 24);
        v20 = *(unsigned int *)(a1 + 32);
        v21 = 8 * v20;
        v16 = v19;
        v17 = &v19[8 * v20 - 8];
        v18 = &v19[8 * v20];
        if ( !v18 )
          goto LABEL_19;
      }
      else
      {
        v16 = *(char **)(a1 + 24);
        v17 = &v16[v15 - 8];
        v18 = &v16[v15];
      }
      *(_QWORD *)v18 = *(_QWORD *)v17;
      v19 = *(char **)(a1 + 24);
      v20 = *(unsigned int *)(a1 + 32);
      v21 = 8 * v20;
      v17 = &v19[8 * v20 - 8];
LABEL_19:
      if ( v16 != v17 )
      {
        v28 = v16;
        memmove(&v19[v21 - (v17 - v16)], v16, v17 - v16);
        LODWORD(v20) = *(_DWORD *)(a1 + 32);
        v16 = v28;
      }
      a6 = (unsigned int)(v20 + 1);
      *(_DWORD *)(a1 + 32) = a6;
      *(_QWORD *)v16 = v13;
LABEL_12:
      v7 = *(unsigned __int8 **)(a1 + 16);
      v11 += 32;
      result = *v7;
    }
    while ( v12 != v11 );
  }
  if ( (_BYTE)result == 84 )
  {
    result = *((_QWORD *)v7 - 1);
    v22 = 32LL * *((unsigned int *)v7 + 18);
    v23 = v22 + 8LL * (*((_DWORD *)v7 + 1) & 0x7FFFFFF);
    v24 = (__int64 *)(result + v22);
    v25 = (__int64 *)(result + v23);
    if ( v25 != v24 )
    {
      result = *(unsigned int *)(a1 + 32);
      do
      {
        v26 = *v24;
        if ( result + 1 > (unsigned __int64)*(unsigned int *)(a1 + 36) )
        {
          sub_C8D5F0(a1 + 24, (const void *)(a1 + 40), result + 1, 8u, a5, a6);
          result = *(unsigned int *)(a1 + 32);
        }
        ++v24;
        *(_QWORD *)(*(_QWORD *)(a1 + 24) + 8 * result) = v26;
        result = (unsigned int)(*(_DWORD *)(a1 + 32) + 1);
        *(_DWORD *)(a1 + 32) = result;
      }
      while ( v25 != v24 );
    }
  }
  return result;
}
