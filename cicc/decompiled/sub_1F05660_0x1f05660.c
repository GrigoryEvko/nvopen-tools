// Function: sub_1F05660
// Address: 0x1f05660
//
__int64 __fastcall sub_1F05660(__int64 a1, __int64 a2)
{
  __int64 v4; // r12
  __int64 v5; // rcx
  __int64 v6; // rbx
  unsigned int v7; // esi
  __int64 v8; // rdx
  __int64 v9; // rax
  __int64 v10; // rax
  __int64 v11; // r9
  int v12; // r8d
  unsigned int v13; // edi
  __int64 v14; // rcx
  __int64 v15; // rax
  int v16; // r10d
  __int64 v17; // rsi
  __m128i *v19; // rax
  __int64 v20; // rdx
  __int64 v21; // rdx
  __m128i v22; // [rsp+0h] [rbp-40h] BYREF
  __int64 v23; // [rsp+10h] [rbp-30h]

  v4 = *(unsigned int *)(a2 + 12);
  v5 = *(unsigned int *)(a1 + 8);
  v6 = *(unsigned __int16 *)(*(_QWORD *)(a1 + 208) + 2 * v4);
  v7 = *(_DWORD *)(a2 + 12);
  if ( (unsigned int)v6 >= (unsigned int)v5 )
  {
LABEL_12:
    v6 = 0xFFFFFFFFLL;
  }
  else
  {
    v8 = *(_QWORD *)a1;
    while ( 1 )
    {
      v9 = v8 + 24LL * (unsigned int)v6;
      if ( v7 == *(_DWORD *)(v9 + 12) )
      {
        v10 = *(unsigned int *)(v9 + 16);
        if ( (_DWORD)v10 != -1 && *(_DWORD *)(v8 + 24 * v10 + 20) == -1 )
          break;
      }
      v6 = (unsigned int)(v6 + 0x10000);
      if ( (unsigned int)v5 <= (unsigned int)v6 )
        goto LABEL_12;
    }
  }
  v11 = *(_QWORD *)a2;
  v12 = *(_DWORD *)(a2 + 8);
  if ( *(_DWORD *)(a1 + 228) )
  {
    v13 = *(_DWORD *)(a1 + 224);
    v14 = 24LL * v13;
    v15 = v14 + *(_QWORD *)a1;
    v16 = *(_DWORD *)(v15 + 20);
    *(_QWORD *)v15 = v11;
    *(_DWORD *)(v15 + 8) = v12;
    *(_DWORD *)(v15 + 12) = v7;
    *(_QWORD *)(v15 + 16) = -1;
    --*(_DWORD *)(a1 + 228);
    *(_DWORD *)(a1 + 224) = v16;
  }
  else
  {
    v22.m128i_i64[0] = *(_QWORD *)a2;
    v22.m128i_i64[1] = __PAIR64__(v7, v12);
    v23 = -1;
    if ( (unsigned int)v5 >= *(_DWORD *)(a1 + 12) )
    {
      sub_16CD150(a1, (const void *)(a1 + 16), 0, 24, v12, v11);
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
  if ( (_DWORD)v6 == -1 )
  {
    *(_WORD *)(*(_QWORD *)(a1 + 208) + 2 * v4) = v13;
    *(_DWORD *)(*(_QWORD *)a1 + v14 + 16) = v13;
  }
  else
  {
    v17 = *(unsigned int *)(*(_QWORD *)a1 + 24 * v6 + 16);
    *(_DWORD *)(*(_QWORD *)a1 + 24 * v17 + 20) = v13;
    *(_DWORD *)(*(_QWORD *)a1 + 24 * v6 + 16) = v13;
    *(_DWORD *)(*(_QWORD *)a1 + v14 + 16) = v17;
  }
  return a1;
}
