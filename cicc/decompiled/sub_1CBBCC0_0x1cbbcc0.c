// Function: sub_1CBBCC0
// Address: 0x1cbbcc0
//
__int64 __fastcall sub_1CBBCC0(__int64 a1, __int64 a2, __int64 a3, __int64 a4, int a5, int a6)
{
  unsigned int v7; // eax
  unsigned int v8; // r13d
  __int64 result; // rax
  unsigned int v10; // r13d
  unsigned __int64 v11; // r14
  unsigned int v12; // r15d
  unsigned int v13; // edx
  unsigned __int64 v14; // rdi
  unsigned __int64 v15; // r15
  __int64 v16; // rax
  int v17; // ecx
  unsigned int v18; // r8d
  int v19; // ecx
  __int64 v20; // rdx
  unsigned __int64 v21; // rax
  unsigned int v22; // r14d
  int v23; // r13d
  unsigned __int64 v24; // rax
  unsigned __int64 v25; // r15
  unsigned int v26; // [rsp+8h] [rbp-38h]
  __int64 v27; // [rsp+8h] [rbp-38h]

  v7 = *(_DWORD *)(a1 + 16);
  v8 = *(_DWORD *)(a2 + 16);
  if ( v7 >= v8 )
    goto LABEL_2;
  v11 = *(_QWORD *)(a1 + 8);
  if ( v8 > v11 << 6 )
  {
    v14 = *(_QWORD *)a1;
    v15 = (v8 + 63) >> 6;
    if ( v15 < 2 * v11 )
      v15 = 2 * v11;
    v16 = (__int64)realloc(v14, 8 * v15, 8 * (int)v15, (_DWORD)v11 << 6, a5, a6);
    if ( !v16 && (8 * v15 || (v16 = malloc(1u)) == 0) )
    {
      v27 = v16;
      sub_16BD1C0("Allocation failed", 1u);
      v16 = v27;
    }
    v17 = *(_DWORD *)(a1 + 16);
    *(_QWORD *)a1 = v16;
    *(_QWORD *)(a1 + 8) = v15;
    v18 = (unsigned int)(v17 + 63) >> 6;
    if ( v15 > v18 )
    {
      v25 = v15 - v18;
      if ( v25 )
      {
        v26 = (unsigned int)(v17 + 63) >> 6;
        memset((void *)(v16 + 8LL * v18), 0, 8 * v25);
        v17 = *(_DWORD *)(a1 + 16);
        v16 = *(_QWORD *)a1;
        v18 = v26;
      }
    }
    v19 = v17 & 0x3F;
    if ( v19 )
    {
      *(_QWORD *)(v16 + 8LL * (v18 - 1)) &= ~(-1LL << v19);
      v16 = *(_QWORD *)a1;
    }
    v20 = *(_QWORD *)(a1 + 8) - (unsigned int)v11;
    if ( v20 )
      memset((void *)(v16 + 8LL * (unsigned int)v11), 0, 8 * v20);
    v7 = *(_DWORD *)(a1 + 16);
    v13 = v7;
    if ( v8 <= v7 )
      goto LABEL_10;
    v11 = *(_QWORD *)(a1 + 8);
  }
  v12 = (v7 + 63) >> 6;
  if ( v11 > v12 )
  {
    memset((void *)(*(_QWORD *)a1 + 8LL * v12), 0, 8 * (v11 - v12));
    v7 = *(_DWORD *)(a1 + 16);
  }
  v13 = v7;
  if ( (v7 & 0x3F) != 0 )
  {
    *(_QWORD *)(*(_QWORD *)a1 + 8LL * (v12 - 1)) &= ~(-1LL << (v7 & 0x3F));
    v13 = *(_DWORD *)(a1 + 16);
  }
LABEL_10:
  *(_DWORD *)(a1 + 16) = v8;
  if ( v8 < v13 )
  {
    v21 = *(_QWORD *)(a1 + 8);
    v22 = (v8 + 63) >> 6;
    if ( v21 > v22 )
    {
      v24 = v21 - v22;
      if ( v24 )
      {
        memset((void *)(*(_QWORD *)a1 + 8LL * v22), 0, 8 * v24);
        v8 = *(_DWORD *)(a1 + 16);
      }
    }
    v23 = v8 & 0x3F;
    if ( v23 )
      *(_QWORD *)(*(_QWORD *)a1 + 8LL * (v22 - 1)) &= ~(-1LL << v23);
  }
  v8 = *(_DWORD *)(a2 + 16);
LABEL_2:
  result = 0;
  v10 = (v8 + 63) >> 6;
  if ( v10 )
  {
    do
    {
      *(_QWORD *)(*(_QWORD *)a1 + 8 * result) |= *(_QWORD *)(*(_QWORD *)a2 + 8 * result);
      ++result;
    }
    while ( v10 != result );
  }
  return result;
}
