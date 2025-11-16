// Function: sub_1E08750
// Address: 0x1e08750
//
__int64 __fastcall sub_1E08750(__int64 a1, __int64 a2, __int64 a3)
{
  __int64 (*v3)(); // rax
  __int64 v5; // rbx
  int v6; // eax
  unsigned int v7; // r15d
  __int64 v8; // rax
  size_t v9; // r9
  size_t v10; // rdx
  void *v11; // r8
  unsigned __int16 *v13; // rdx
  unsigned __int64 v14; // rcx
  unsigned int *v15; // r8
  unsigned int *i; // r9
  unsigned __int64 v17; // rcx
  __int64 v18; // rsi
  __int64 v19; // rdx
  int v20; // eax
  __int64 v21; // rax
  void *v22; // [rsp+8h] [rbp-48h]
  size_t v23; // [rsp+10h] [rbp-40h]
  size_t n; // [rsp+18h] [rbp-38h]
  size_t na; // [rsp+18h] [rbp-38h]

  v3 = *(__int64 (**)())(**(_QWORD **)(a3 + 16) + 112LL);
  if ( v3 == sub_1D00B10 )
    BUG();
  v5 = v3();
  v6 = *(_DWORD *)(v5 + 16);
  *(_QWORD *)a1 = 0;
  *(_QWORD *)(a1 + 8) = 0;
  *(_DWORD *)(a1 + 16) = v6;
  v7 = (unsigned int)(v6 + 63) >> 6;
  n = 8LL * v7;
  v8 = malloc(n);
  v9 = n;
  v10 = v7;
  v11 = (void *)v8;
  if ( !v8 )
  {
    if ( n || (v21 = malloc(1u), v10 = v7, v9 = 0, v11 = 0, !v21) )
    {
      v22 = v11;
      v23 = v9;
      na = v10;
      sub_16BD1C0("Allocation failed", 1u);
      v10 = na;
      v9 = v23;
      v11 = v22;
    }
    else
    {
      v11 = (void *)v21;
    }
  }
  *(_QWORD *)a1 = v11;
  *(_QWORD *)(a1 + 8) = v10;
  if ( v7 )
  {
    memset(v11, 0, v9);
    if ( !*(_BYTE *)(a2 + 104) )
      return a1;
  }
  else if ( !*(_BYTE *)(a2 + 104) )
  {
    return a1;
  }
  v13 = (unsigned __int16 *)sub_1E6A620(*(_QWORD *)(a3 + 40));
  if ( v13 )
  {
    while ( 1 )
    {
      v14 = *v13;
      if ( !(_WORD)v14 )
        break;
      ++v13;
      *(_QWORD *)(*(_QWORD *)a1 + ((v14 >> 3) & 0x1FF8)) |= 1LL << v14;
    }
  }
  v15 = *(unsigned int **)(a2 + 80);
  for ( i = *(unsigned int **)(a2 + 88); i != v15; v15 += 3 )
  {
    v17 = *v15;
    v18 = *(_QWORD *)(v5 + 56) + 2LL * *(unsigned int *)(*(_QWORD *)(v5 + 8) + 24 * v17 + 4);
LABEL_14:
    v19 = v18;
    while ( v19 )
    {
      v19 += 2;
      *(_QWORD *)(*(_QWORD *)a1 + ((v17 >> 3) & 0x1FF8)) &= ~(1LL << v17);
      v20 = *(unsigned __int16 *)(v19 - 2);
      v18 = 0;
      v17 = (unsigned int)(v20 + v17);
      if ( !(_WORD)v20 )
        goto LABEL_14;
    }
  }
  return a1;
}
