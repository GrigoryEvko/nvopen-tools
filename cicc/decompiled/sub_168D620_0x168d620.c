// Function: sub_168D620
// Address: 0x168d620
//
__int64 __fastcall sub_168D620(__int64 *a1, unsigned __int8 *a2, size_t a3, int a4)
{
  size_t v4; // r9
  int v5; // r8d
  __int64 v7; // rbx
  __int64 v8; // rax
  __int64 v9; // rdx
  unsigned int v10; // r13d
  unsigned __int64 v11; // r12
  __int64 v12; // rax
  __int64 v13; // rdx
  unsigned __int64 v14; // r13
  unsigned __int64 v15; // r12
  __int64 result; // rax
  _BYTE *v17; // rdi
  __int64 v18; // r13
  char *v19; // rsi
  __int64 v20; // rax
  int v21; // [rsp+Ch] [rbp-54h]
  size_t v23; // [rsp+10h] [rbp-50h]
  int v24; // [rsp+18h] [rbp-48h]
  int v25; // [rsp+18h] [rbp-48h]
  __int64 v26; // [rsp+18h] [rbp-48h]
  _QWORD v27[7]; // [rsp+28h] [rbp-38h] BYREF

  v4 = a3;
  v5 = a4;
  v7 = *a1;
  v8 = *(_QWORD *)(*a1 + 8);
  v9 = *(_QWORD *)(*a1 + 16);
  *(_QWORD *)(*a1 + 88) += 40LL;
  if ( ((v8 + 7) & 0xFFFFFFFFFFFFFFF8LL) - v8 + 40 <= v9 - v8 )
  {
    v15 = (v8 + 7) & 0xFFFFFFFFFFFFFFF8LL;
    *(_QWORD *)(v7 + 8) = v15 + 40;
    v14 = v15;
  }
  else
  {
    v10 = *(_DWORD *)(v7 + 32);
    v11 = 0x40000000000LL;
    if ( v10 >> 7 < 0x1E )
      v11 = 4096LL << (v10 >> 7);
    v12 = malloc(v11);
    v4 = a3;
    v5 = a4;
    if ( !v12 )
    {
      sub_16BD1C0("Allocation failed");
      v10 = *(_DWORD *)(v7 + 32);
      v5 = a4;
      v4 = a3;
      v12 = 0;
    }
    if ( v10 >= *(_DWORD *)(v7 + 36) )
    {
      v21 = v5;
      v23 = v4;
      v26 = v12;
      sub_16CD150(v7 + 24, v7 + 40, 0, 8);
      v10 = *(_DWORD *)(v7 + 32);
      v5 = v21;
      v4 = v23;
      v12 = v26;
    }
    v13 = v10;
    v14 = (v12 + 7) & 0xFFFFFFFFFFFFFFF8LL;
    *(_QWORD *)(*(_QWORD *)(v7 + 24) + 8 * v13) = v12;
    ++*(_DWORD *)(v7 + 32);
    *(_QWORD *)(v7 + 8) = v14 + 40;
    *(_QWORD *)(v7 + 16) = v12 + v11;
    v15 = v14;
  }
  result = v15 + 16;
  *(_QWORD *)v15 = v15 + 16;
  if ( a2 )
  {
    v27[0] = a3;
    if ( a3 > 0xF )
    {
      v24 = v5;
      v20 = sub_22409D0(v15, v27, 0);
      v5 = v24;
      *(_QWORD *)v15 = v20;
      v17 = (_BYTE *)v20;
      *(_QWORD *)(v15 + 16) = v27[0];
    }
    else
    {
      v17 = *(_BYTE **)v15;
      if ( a3 == 1 )
      {
        result = *a2;
        *v17 = result;
        v4 = v27[0];
        v17 = *(_BYTE **)v15;
LABEL_13:
        *(_QWORD *)(v15 + 8) = v4;
        v17[v4] = 0;
        goto LABEL_15;
      }
      if ( !a3 )
        goto LABEL_13;
    }
    v25 = v5;
    result = (__int64)memcpy(v17, a2, a3);
    v4 = v27[0];
    v17 = *(_BYTE **)v15;
    v5 = v25;
    goto LABEL_13;
  }
  *(_QWORD *)(v15 + 8) = 0;
  *(_BYTE *)(v15 + 16) = 0;
LABEL_15:
  v18 = v14 | 4;
  *(_DWORD *)(v15 + 32) = v5;
  v27[0] = v18;
  v19 = *(char **)(v7 + 120);
  if ( v19 == *(char **)(v7 + 128) )
    return sub_168D4A0((char **)(v7 + 112), v19, v27);
  if ( v19 )
  {
    *(_QWORD *)v19 = v18;
    v19 = *(char **)(v7 + 120);
  }
  *(_QWORD *)(v7 + 120) = v19 + 8;
  return result;
}
