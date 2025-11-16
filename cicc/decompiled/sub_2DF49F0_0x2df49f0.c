// Function: sub_2DF49F0
// Address: 0x2df49f0
//
__int64 __fastcall sub_2DF49F0(__int64 a1, const __m128i *a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  const __m128i *v6; // r15
  __int32 v8; // esi
  unsigned int v9; // r12d
  __int64 v10; // rbx
  unsigned __int64 v11; // rdi
  unsigned __int64 v12; // rax
  unsigned __int64 v14; // rdx
  __m128i *v15; // rax
  __int64 v16; // rdx
  __int64 v17; // rax
  unsigned int v18; // r12d
  _BYTE *v19; // rdi
  char v20; // al
  __int64 v21; // r14
  __int64 v22; // r8
  const void *v23; // rsi
  __int8 *v24; // r15

  v6 = a2;
  if ( !a2->m128i_i8[0] )
  {
    v8 = a2->m128i_i32[2];
    v9 = -1;
    if ( !v8 )
      return v9;
    v10 = *(unsigned int *)(a1 + 64);
    v11 = *(_QWORD *)(a1 + 56);
    if ( (_DWORD)v10 )
    {
      v12 = v11;
      v9 = 0;
      while ( *(_BYTE *)v12
           || *(_DWORD *)(v12 + 8) != v8
           || (((unsigned __int32)v6->m128i_i32[0] >> 8) & 0xFFF) != ((*(_DWORD *)v12 >> 8) & 0xFFF) )
      {
        ++v9;
        v12 += 40LL;
        if ( v9 == (_DWORD)v10 )
          goto LABEL_12;
      }
      return v9;
    }
LABEL_12:
    v14 = v10 + 1;
    if ( v10 + 1 > (unsigned __int64)*(unsigned int *)(a1 + 68) )
    {
      v22 = a1 + 56;
      v23 = (const void *)(a1 + 72);
      if ( v11 > (unsigned __int64)v6 || (unsigned __int64)v6 >= v11 + 40 * v10 )
      {
        sub_C8D5F0(a1 + 56, v23, v14, 0x28u, v22, a6);
        v11 = *(_QWORD *)(a1 + 56);
        v10 = *(unsigned int *)(a1 + 64);
      }
      else
      {
        v24 = &v6->m128i_i8[-v11];
        sub_C8D5F0(a1 + 56, v23, v14, 0x28u, v22, a6);
        v11 = *(_QWORD *)(a1 + 56);
        v10 = *(unsigned int *)(a1 + 64);
        v6 = (const __m128i *)&v24[v11];
      }
    }
    v15 = (__m128i *)(v11 + 40 * v10);
    *v15 = _mm_loadu_si128(v6);
    v15[1] = _mm_loadu_si128(v6 + 1);
    v15[2].m128i_i64[0] = v6[2].m128i_i64[0];
    v16 = *(_QWORD *)(a1 + 56);
    v17 = (unsigned int)(*(_DWORD *)(a1 + 64) + 1);
    *(_DWORD *)(a1 + 64) = v17;
    *(_QWORD *)(v16 + 40 * v17 - 24) = 0;
    v18 = *(_DWORD *)(a1 + 64);
    v19 = (_BYTE *)(*(_QWORD *)(a1 + 56) + 40LL * v18 - 40);
    if ( !*v19 )
    {
      v20 = v19[3];
      if ( (v20 & 0x10) != 0 )
      {
        v19[3] = v20 & 0xBF;
        v19 = (_BYTE *)(*(_QWORD *)(a1 + 56) + 40LL * *(unsigned int *)(a1 + 64) - 40);
      }
      sub_2EAB250(v19, 0);
      v18 = *(_DWORD *)(a1 + 64);
    }
    return v18 - 1;
  }
  v10 = *(unsigned int *)(a1 + 64);
  v21 = 0;
  v9 = 0;
  if ( !(_DWORD)v10 )
  {
LABEL_11:
    v11 = *(_QWORD *)(a1 + 56);
    goto LABEL_12;
  }
  while ( !(unsigned __int8)sub_2EAB6C0(a2, v21 + *(_QWORD *)(a1 + 56)) )
  {
    ++v9;
    v21 += 40;
    if ( v9 == (_DWORD)v10 )
    {
      v10 = *(unsigned int *)(a1 + 64);
      goto LABEL_11;
    }
  }
  return v9;
}
