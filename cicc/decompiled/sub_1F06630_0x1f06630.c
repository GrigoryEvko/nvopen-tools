// Function: sub_1F06630
// Address: 0x1f06630
//
__int64 __fastcall sub_1F06630(__int64 a1, __int64 a2)
{
  int v4; // r9d
  __int64 v5; // rax
  __int64 v6; // rcx
  unsigned int v7; // edi
  __int64 v8; // r12
  __int64 v9; // rbx
  __int64 v10; // rsi
  _DWORD *v11; // rax
  __int64 v12; // rax
  unsigned int v13; // r10d
  __int64 v14; // rdi
  __int32 v15; // esi
  unsigned int v16; // ecx
  __int64 v17; // r8
  __int64 v18; // rax
  int v19; // r11d
  __int64 v20; // rbx
  __int64 v21; // rsi
  __m128i v23; // xmm1
  __int64 v24; // rcx
  __int64 v25; // r8
  __m128i v26; // [rsp+0h] [rbp-40h] BYREF
  __m128i v27; // [rsp+10h] [rbp-30h] BYREF

  v4 = *(_DWORD *)a2;
  v5 = *(_QWORD *)(a1 + 272);
  v6 = *(unsigned int *)(a1 + 8);
  v7 = *(_DWORD *)a2 & 0x7FFFFFFF;
  v8 = v7;
  v9 = *(unsigned __int8 *)(v5 + v7);
  if ( (unsigned int)v9 >= (unsigned int)v6 )
  {
LABEL_12:
    v9 = 0xFFFFFFFFLL;
  }
  else
  {
    v10 = *(_QWORD *)a1;
    while ( 1 )
    {
      v11 = (_DWORD *)(v10 + 32LL * (unsigned int)v9);
      if ( v7 == (*v11 & 0x7FFFFFFF) )
      {
        v12 = (unsigned int)v11[6];
        if ( (_DWORD)v12 != -1 && *(_DWORD *)(v10 + 32 * v12 + 28) == -1 )
          break;
      }
      v9 = (unsigned int)(v9 + 256);
      if ( (unsigned int)v6 <= (unsigned int)v9 )
        goto LABEL_12;
    }
  }
  v13 = *(_DWORD *)(a2 + 4);
  v14 = *(_QWORD *)(a2 + 8);
  v15 = *(_DWORD *)(a2 + 16);
  if ( *(_DWORD *)(a1 + 292) )
  {
    v16 = *(_DWORD *)(a1 + 288);
    v17 = 32LL * v16;
    v18 = v17 + *(_QWORD *)a1;
    v19 = *(_DWORD *)(v18 + 28);
    *(_DWORD *)v18 = v4;
    *(_DWORD *)(v18 + 4) = v13;
    *(_QWORD *)(v18 + 8) = v14;
    *(_DWORD *)(v18 + 16) = v15;
    *(_QWORD *)(v18 + 24) = -1;
    --*(_DWORD *)(a1 + 292);
    *(_DWORD *)(a1 + 288) = v19;
  }
  else
  {
    v26.m128i_i64[0] = __PAIR64__(v13, v4);
    v26.m128i_i64[1] = v14;
    v27.m128i_i32[0] = v15;
    v27.m128i_i64[1] = -1;
    if ( (unsigned int)v6 >= *(_DWORD *)(a1 + 12) )
    {
      sub_16CD150(a1, (const void *)(a1 + 16), 0, 32, a2, v4);
      v6 = *(unsigned int *)(a1 + 8);
    }
    v23 = _mm_loadu_si128(&v27);
    v24 = *(_QWORD *)a1 + 32 * v6;
    *(__m128i *)v24 = _mm_loadu_si128(&v26);
    *(__m128i *)(v24 + 16) = v23;
    v25 = *(unsigned int *)(a1 + 8);
    v16 = *(_DWORD *)(a1 + 8);
    *(_DWORD *)(a1 + 8) = v16 + 1;
    v17 = 32 * v25;
  }
  if ( (_DWORD)v9 == -1 )
  {
    *(_BYTE *)(*(_QWORD *)(a1 + 272) + v8) = v16;
    *(_DWORD *)(*(_QWORD *)a1 + v17 + 24) = v16;
  }
  else
  {
    v20 = 32 * v9;
    v21 = *(unsigned int *)(*(_QWORD *)a1 + v20 + 24);
    *(_DWORD *)(*(_QWORD *)a1 + 32 * v21 + 28) = v16;
    *(_DWORD *)(*(_QWORD *)a1 + v20 + 24) = v16;
    *(_DWORD *)(*(_QWORD *)a1 + v17 + 24) = v21;
  }
  return a1;
}
