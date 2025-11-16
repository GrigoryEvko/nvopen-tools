// Function: sub_2F932E0
// Address: 0x2f932e0
//
__int64 __fastcall sub_2F932E0(__int64 a1, __int64 a2)
{
  __int64 v4; // r12
  __int64 v5; // rcx
  unsigned __int64 v6; // rdx
  __int64 v7; // rbx
  int v8; // esi
  __int64 v9; // rax
  __int64 v10; // rax
  __int64 v11; // r9
  int v12; // r8d
  unsigned int v13; // edi
  __int64 v14; // rcx
  __int64 v15; // rdx
  int v16; // r10d
  __int64 v17; // rsi
  unsigned __int64 v19; // rax
  unsigned __int64 v20; // r8
  const __m128i *v21; // r14
  __m128i *v22; // rax
  const void *v23; // rsi
  char *v24; // r14
  __int64 v25; // [rsp+0h] [rbp-40h] BYREF
  int v26; // [rsp+8h] [rbp-38h]
  int v27; // [rsp+Ch] [rbp-34h]
  __int64 v28; // [rsp+10h] [rbp-30h]

  v4 = *(unsigned int *)(a2 + 12);
  v5 = *(unsigned int *)(a1 + 8);
  v6 = *(_QWORD *)a1;
  v7 = *(unsigned __int16 *)(*(_QWORD *)(a1 + 208) + 2 * v4);
  v8 = *(_DWORD *)(a2 + 12);
  if ( (unsigned int)v7 < (unsigned int)v5 )
  {
    while ( 1 )
    {
      v9 = v6 + 24LL * (unsigned int)v7;
      if ( v8 == *(_DWORD *)(v9 + 12) )
      {
        v10 = *(unsigned int *)(v9 + 16);
        if ( (_DWORD)v10 != -1 && *(_DWORD *)(v6 + 24 * v10 + 20) == -1 )
          break;
      }
      v7 = (unsigned int)(v7 + 0x10000);
      if ( (unsigned int)v5 <= (unsigned int)v7 )
        goto LABEL_12;
    }
  }
  else
  {
LABEL_12:
    v7 = 0xFFFFFFFFLL;
  }
  v11 = *(_QWORD *)a2;
  v12 = *(_DWORD *)(a2 + 8);
  if ( *(_DWORD *)(a1 + 228) )
  {
    v13 = *(_DWORD *)(a1 + 224);
    v14 = 24LL * v13;
    v15 = v14 + v6;
    v16 = *(_DWORD *)(v15 + 20);
    *(_QWORD *)v15 = v11;
    *(_DWORD *)(v15 + 8) = v12;
    *(_DWORD *)(v15 + 12) = v8;
    *(_QWORD *)(v15 + 16) = -1;
    --*(_DWORD *)(a1 + 228);
    *(_DWORD *)(a1 + 224) = v16;
  }
  else
  {
    v19 = *(unsigned int *)(a1 + 12);
    v26 = *(_DWORD *)(a2 + 8);
    v20 = v5 + 1;
    v21 = (const __m128i *)&v25;
    v25 = v11;
    v27 = v8;
    v28 = -1;
    if ( v5 + 1 > v19 )
    {
      v23 = (const void *)(a1 + 16);
      if ( v6 > (unsigned __int64)&v25 || (unsigned __int64)&v25 >= v6 + 24 * v5 )
      {
        sub_C8D5F0(a1, v23, v20, 0x18u, v20, v11);
        v6 = *(_QWORD *)a1;
        v5 = *(unsigned int *)(a1 + 8);
      }
      else
      {
        v24 = (char *)&v25 - v6;
        sub_C8D5F0(a1, v23, v20, 0x18u, v20, v11);
        v6 = *(_QWORD *)a1;
        v5 = *(unsigned int *)(a1 + 8);
        v21 = (const __m128i *)&v24[*(_QWORD *)a1];
      }
    }
    v22 = (__m128i *)(v6 + 24 * v5);
    *v22 = _mm_loadu_si128(v21);
    v22[1].m128i_i64[0] = v21[1].m128i_i64[0];
    v13 = *(_DWORD *)(a1 + 8);
    *(_DWORD *)(a1 + 8) = v13 + 1;
    v14 = 24LL * v13;
  }
  if ( (_DWORD)v7 == -1 )
  {
    *(_WORD *)(*(_QWORD *)(a1 + 208) + 2 * v4) = v13;
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
