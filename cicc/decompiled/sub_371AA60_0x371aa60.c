// Function: sub_371AA60
// Address: 0x371aa60
//
__int64 __fastcall sub_371AA60(unsigned __int64 ***a1, unsigned __int64 a2, char a3)
{
  unsigned __int64 v3; // r9
  unsigned __int64 *v5; // rsi
  unsigned __int64 *v6; // rdx
  unsigned __int64 v7; // r13
  unsigned __int64 *v9; // rdi
  unsigned __int64 v10; // r11
  unsigned __int64 **v11; // r12
  unsigned __int64 *v12; // rcx
  unsigned __int64 v13; // rax
  const __m128i *v14; // rbx
  const __m128i *v15; // rdx
  __int64 v16; // rax
  __int64 **v17; // r14
  __int64 *v18; // r12
  __int64 v19; // rax
  unsigned __int64 v20; // rdx
  unsigned __int64 v21; // r8
  __m128i *v22; // rax
  const void *v24; // rsi
  __int8 *v25; // rbx
  unsigned __int64 *v26; // r12
  __int64 v27; // rax
  __int64 v28; // r12
  unsigned int v29; // eax
  __int64 v30; // rdx

  v3 = a2;
  v5 = **a1;
  v6 = &v5[3 * *((unsigned int *)*a1 + 2)];
  v7 = (unsigned __int64)*a1[1];
  if ( v5 == v6 )
    return 0;
  v9 = **a1;
  do
  {
    if ( (v7 & ~(-1LL << *((_BYTE *)v9 + 16))) == 0 )
      break;
    v9 += 3;
  }
  while ( v6 != v9 );
  v10 = v7 - 1;
  while ( v6 == v9 )
  {
LABEL_21:
    if ( v5 == v9 )
      return 0;
    v26 = v9 - 3;
    v7 = -(1LL << *((_BYTE *)v9 - 8)) & ((1LL << *((_BYTE *)v9 - 8)) + v10);
    if ( v7 >= v3 )
    {
      if ( a3 )
        return 0;
    }
    for ( ; v5 != v26; v26 -= 3 )
    {
      if ( v7 != (((1LL << *((_BYTE *)v26 - 8)) + v10) & -(1LL << *((_BYTE *)v26 - 8))) )
        break;
    }
    v6 = v9;
    v9 = v26;
  }
  v11 = a1[2];
  if ( !a3 )
  {
    v14 = (const __m128i *)v9[1];
    goto LABEL_30;
  }
  v12 = v9;
  v13 = v3 - v7;
  while ( *v12 > v13 )
  {
    v12 += 3;
    if ( v6 == v12 )
      goto LABEL_21;
  }
  v14 = (const __m128i *)v12[1];
  if ( v14->m128i_i64[1] <= v13 )
  {
    v9 = v12;
LABEL_30:
    v27 = v14[1].m128i_i64[1];
    v17 = (__int64 **)*v11;
    if ( v27 )
    {
      v9[1] = v27;
    }
    else
    {
      v28 = **v17;
      v29 = *(_DWORD *)(v28 + 8);
      v30 = *(_QWORD *)v28 + 24LL * v29;
      if ( (unsigned __int64 *)v30 != v9 + 3 )
      {
        memmove(v9, v9 + 3, v30 - (_QWORD)(v9 + 3));
        v29 = *(_DWORD *)(v28 + 8);
      }
      *(_DWORD *)(v28 + 8) = v29 - 1;
    }
    goto LABEL_15;
  }
  do
  {
    v15 = v14;
    v14 = (const __m128i *)v14[1].m128i_i64[1];
  }
  while ( v14->m128i_i64[1] > v13 );
  v16 = v14[1].m128i_i64[1];
  v17 = (__int64 **)*v11;
  v15[1].m128i_i64[1] = v16;
  if ( !v16 )
    *v12 = v15->m128i_u64[1];
LABEL_15:
  v18 = v17[1];
  v19 = *((unsigned int *)v18 + 2);
  v20 = *v18;
  v21 = v19 + 1;
  if ( v19 + 1 > (unsigned __int64)*((unsigned int *)v18 + 3) )
  {
    v24 = v18 + 2;
    if ( v20 > (unsigned __int64)v14 || (unsigned __int64)v14 >= v20 + 40 * v19 )
    {
      sub_C8D5F0((__int64)v17[1], v24, v21, 0x28u, v21, v3);
      v20 = *v18;
      v19 = *((unsigned int *)v18 + 2);
    }
    else
    {
      v25 = &v14->m128i_i8[-v20];
      sub_C8D5F0((__int64)v17[1], v24, v21, 0x28u, v21, v3);
      v20 = *v18;
      v19 = *((unsigned int *)v18 + 2);
      v14 = (const __m128i *)&v25[*v18];
    }
  }
  v22 = (__m128i *)(v20 + 40 * v19);
  *v22 = _mm_loadu_si128(v14);
  v22[1] = _mm_loadu_si128(v14 + 1);
  v22[2].m128i_i64[0] = v14[2].m128i_i64[0];
  ++*((_DWORD *)v18 + 2);
  *(_QWORD *)(*v17[1] + 40LL * *((unsigned int *)v17[1] + 2) - 40) = v7;
  *v17[2] = *(_QWORD *)(*v17[1] + 40LL * *((unsigned int *)v17[1] + 2) - 32)
          + *(_QWORD *)(*v17[1] + 40LL * *((unsigned int *)v17[1] + 2) - 40);
  return 1;
}
