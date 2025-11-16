// Function: sub_1E74F70
// Address: 0x1e74f70
//
__int64 __fastcall sub_1E74F70(__int64 a1, __int64 a2)
{
  int v4; // r9d
  __int64 v5; // rsi
  __int64 v6; // r12
  __int64 v7; // rbx
  __int64 v8; // rcx
  _DWORD *v9; // rdx
  __int64 v10; // rax
  unsigned int v11; // r10d
  __int64 v12; // rdi
  unsigned int v13; // esi
  __int64 v14; // rcx
  __int64 v15; // rax
  int v16; // r8d
  __int64 v17; // rdi
  __m128i *v19; // rax
  __int64 v20; // rdx
  __int64 v21; // rdx
  __m128i v22; // [rsp+0h] [rbp-40h] BYREF
  __int64 v23; // [rsp+10h] [rbp-30h]

  v4 = *(_DWORD *)a2;
  v5 = *(unsigned int *)(a1 + 8);
  v6 = v4 & 0x7FFFFFFF;
  v7 = *(unsigned __int8 *)(*(_QWORD *)(a1 + 208) + v6);
  if ( (unsigned int)v7 >= (unsigned int)v5 )
  {
LABEL_12:
    v7 = 0xFFFFFFFFLL;
  }
  else
  {
    v8 = *(_QWORD *)a1;
    while ( 1 )
    {
      v9 = (_DWORD *)(v8 + 24LL * (unsigned int)v7);
      if ( (v4 & 0x7FFFFFFF) == (*v9 & 0x7FFFFFFF) )
      {
        v10 = (unsigned int)v9[4];
        if ( (_DWORD)v10 != -1 && *(_DWORD *)(v8 + 24 * v10 + 20) == -1 )
          break;
      }
      v7 = (unsigned int)(v7 + 256);
      if ( (unsigned int)v5 <= (unsigned int)v7 )
        goto LABEL_12;
    }
  }
  v11 = *(_DWORD *)(a2 + 4);
  v12 = *(_QWORD *)(a2 + 8);
  if ( *(_DWORD *)(a1 + 228) )
  {
    v13 = *(_DWORD *)(a1 + 224);
    v14 = 24LL * v13;
    v15 = v14 + *(_QWORD *)a1;
    v16 = *(_DWORD *)(v15 + 20);
    *(_DWORD *)v15 = v4;
    *(_DWORD *)(v15 + 4) = v11;
    *(_QWORD *)(v15 + 8) = v12;
    *(_QWORD *)(v15 + 16) = -1;
    --*(_DWORD *)(a1 + 228);
    *(_DWORD *)(a1 + 224) = v16;
  }
  else
  {
    v22.m128i_i64[0] = __PAIR64__(v11, v4);
    v22.m128i_i64[1] = v12;
    v23 = -1;
    if ( (unsigned int)v5 >= *(_DWORD *)(a1 + 12) )
    {
      sub_16CD150(a1, (const void *)(a1 + 16), 0, 24, a2, v4);
      v5 = *(unsigned int *)(a1 + 8);
    }
    v19 = (__m128i *)(*(_QWORD *)a1 + 24 * v5);
    v20 = v23;
    *v19 = _mm_loadu_si128(&v22);
    v19[1].m128i_i64[0] = v20;
    v21 = *(unsigned int *)(a1 + 8);
    v13 = *(_DWORD *)(a1 + 8);
    *(_DWORD *)(a1 + 8) = v21 + 1;
    v14 = 24 * v21;
  }
  if ( (_DWORD)v7 == -1 )
  {
    *(_BYTE *)(*(_QWORD *)(a1 + 208) + v6) = v13;
    *(_DWORD *)(*(_QWORD *)a1 + v14 + 16) = v13;
  }
  else
  {
    v17 = *(unsigned int *)(*(_QWORD *)a1 + 24 * v7 + 16);
    *(_DWORD *)(*(_QWORD *)a1 + 24 * v17 + 20) = v13;
    *(_DWORD *)(*(_QWORD *)a1 + 24 * v7 + 16) = v13;
    *(_DWORD *)(*(_QWORD *)a1 + v14 + 16) = v17;
  }
  return a1;
}
