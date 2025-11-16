// Function: sub_1DA81D0
// Address: 0x1da81d0
//
__int64 __fastcall sub_1DA81D0(__int64 a1, const __m128i *a2, __int64 a3, __int64 a4, int a5, int a6)
{
  __int32 v8; // esi
  unsigned int v9; // r12d
  __int64 v10; // rbx
  __int64 v11; // rax
  __m128i *v13; // rax
  __int64 v14; // rdx
  __int64 v15; // rax
  unsigned int v16; // r12d
  _BYTE *v17; // rdi
  char v18; // al
  __int64 v19; // r14

  if ( !a2->m128i_i8[0] )
  {
    v8 = a2->m128i_i32[2];
    v9 = -1;
    if ( !v8 )
      return v9;
    v10 = *(unsigned int *)(a1 + 48);
    if ( (_DWORD)v10 )
    {
      v11 = *(_QWORD *)(a1 + 40);
      v9 = 0;
      while ( *(_BYTE *)v11
           || v8 != *(_DWORD *)(v11 + 8)
           || (((unsigned __int32)a2->m128i_i32[0] >> 8) & 0xFFF) != ((*(_DWORD *)v11 >> 8) & 0xFFF) )
      {
        ++v9;
        v11 += 40;
        if ( (_DWORD)v10 == v9 )
          goto LABEL_11;
      }
      return v9;
    }
LABEL_11:
    if ( *(_DWORD *)(a1 + 52) <= (unsigned int)v10 )
    {
      sub_16CD150(a1 + 40, (const void *)(a1 + 56), 0, 40, a5, a6);
      v10 = *(unsigned int *)(a1 + 48);
    }
    v13 = (__m128i *)(*(_QWORD *)(a1 + 40) + 40 * v10);
    *v13 = _mm_loadu_si128(a2);
    v13[1] = _mm_loadu_si128(a2 + 1);
    v13[2].m128i_i64[0] = a2[2].m128i_i64[0];
    v14 = *(_QWORD *)(a1 + 40);
    v15 = (unsigned int)(*(_DWORD *)(a1 + 48) + 1);
    *(_DWORD *)(a1 + 48) = v15;
    *(_QWORD *)(v14 + 40 * v15 - 24) = 0;
    v16 = *(_DWORD *)(a1 + 48);
    v17 = (_BYTE *)(*(_QWORD *)(a1 + 40) + 40LL * v16 - 40);
    if ( !*v17 )
    {
      v18 = v17[3];
      if ( (v18 & 0x10) != 0 )
      {
        v17[3] = v18 & 0xBF;
        v17 = (_BYTE *)(*(_QWORD *)(a1 + 40) + 40LL * *(unsigned int *)(a1 + 48) - 40);
      }
      sub_1E31260(v17, 0);
      v16 = *(_DWORD *)(a1 + 48);
    }
    return v16 - 1;
  }
  v10 = *(unsigned int *)(a1 + 48);
  if ( !(_DWORD)v10 )
    goto LABEL_11;
  v19 = 0;
  v9 = 0;
  while ( !(unsigned __int8)sub_1E31610(a2, v19 + *(_QWORD *)(a1 + 40)) )
  {
    ++v9;
    v19 += 40;
    if ( v9 == (_DWORD)v10 )
    {
      v10 = *(unsigned int *)(a1 + 48);
      goto LABEL_11;
    }
  }
  return v9;
}
