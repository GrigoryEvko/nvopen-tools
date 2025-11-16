// Function: sub_2F75570
// Address: 0x2f75570
//
__int64 __fastcall sub_2F75570(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  __int64 v7; // rbx
  __int64 v8; // r13
  unsigned __int64 v9; // rdx
  unsigned int *v10; // rbx
  __int64 result; // rax
  __int64 v12; // rdi
  unsigned int *i; // r14
  unsigned int v14; // ecx
  unsigned int v15; // edx
  bool v16; // cf
  unsigned __int64 v17; // rcx
  unsigned __int64 v18; // rdx
  unsigned __int64 v19; // r8
  unsigned __int64 v20; // r10
  unsigned __int64 v21; // r11
  int v22; // esi
  unsigned __int64 v23; // r9
  unsigned __int64 v24; // rdx
  const __m128i *v25; // rax
  __m128i *v26; // rdx
  const void *v27; // rsi
  char *v28; // [rsp+8h] [rbp-58h]
  int v29; // [rsp+10h] [rbp-50h] BYREF
  unsigned __int64 v30; // [rsp+18h] [rbp-48h]
  __int64 v31; // [rsp+20h] [rbp-40h]

  v7 = *(_QWORD *)(a1 + 48);
  if ( *(_BYTE *)(a1 + 56) )
  {
    *(_QWORD *)(v7 + 440) = sub_2F75400((_QWORD *)a1);
    v8 = *(_QWORD *)(a1 + 48);
    v9 = *(unsigned int *)(a1 + 104);
    if ( *(_DWORD *)(v8 + 36) >= (unsigned int)v9 )
      goto LABEL_3;
  }
  else
  {
    *(_QWORD *)(v7 + 440) = *(_QWORD *)(a1 + 64);
    v8 = *(_QWORD *)(a1 + 48);
    v9 = *(unsigned int *)(a1 + 104);
    if ( *(_DWORD *)(v8 + 36) >= (unsigned int)v9 )
      goto LABEL_3;
  }
  sub_C8D5F0(v8 + 24, (const void *)(v8 + 40), v9, 0x18u, a5, a6);
  v8 = *(_QWORD *)(a1 + 48);
  v9 = *(unsigned int *)(a1 + 104);
LABEL_3:
  v10 = *(unsigned int **)(a1 + 96);
  result = 3 * v9;
  v12 = v8 + 24;
  for ( i = &v10[6 * v9]; i != v10; v10 += 6 )
  {
    result = *v10;
    v14 = *(_DWORD *)(a1 + 320);
    v15 = (*v10 - v14) | 0x80000000;
    v16 = (unsigned int)result < v14;
    v17 = *((_QWORD *)v10 + 1);
    if ( !v16 )
      result = v15;
    v18 = *((_QWORD *)v10 + 2);
    if ( __PAIR128__(v17, v18) != 0 )
    {
      v19 = *(unsigned int *)(v8 + 32);
      v20 = *(_QWORD *)(v8 + 24);
      v21 = *(unsigned int *)(v8 + 36);
      v22 = *(_DWORD *)(v8 + 32);
      v23 = v20 + 24 * v19;
      if ( v19 >= v21 )
      {
        v31 = *((_QWORD *)v10 + 2);
        v24 = v19 + 1;
        v29 = result;
        v25 = (const __m128i *)&v29;
        v30 = v17;
        if ( v21 < v19 + 1 )
        {
          v27 = (const void *)(v8 + 40);
          if ( v20 > (unsigned __int64)&v29 || v23 <= (unsigned __int64)&v29 )
          {
            sub_C8D5F0(v12, v27, v24, 0x18u, v19, v23);
            v20 = *(_QWORD *)(v8 + 24);
            v19 = *(unsigned int *)(v8 + 32);
            v25 = (const __m128i *)&v29;
          }
          else
          {
            v28 = (char *)&v29 - v20;
            sub_C8D5F0(v12, v27, v24, 0x18u, v19, v23);
            v20 = *(_QWORD *)(v8 + 24);
            v19 = *(unsigned int *)(v8 + 32);
            v25 = (const __m128i *)&v28[v20];
          }
        }
        v26 = (__m128i *)(v20 + 24 * v19);
        *v26 = _mm_loadu_si128(v25);
        result = v25[1].m128i_i64[0];
        v26[1].m128i_i64[0] = result;
        ++*(_DWORD *)(v8 + 32);
      }
      else
      {
        if ( v23 )
        {
          *(_DWORD *)v23 = result;
          *(_QWORD *)(v23 + 8) = v17;
          *(_QWORD *)(v23 + 16) = v18;
          v22 = *(_DWORD *)(v8 + 32);
        }
        *(_DWORD *)(v8 + 32) = v22 + 1;
      }
    }
  }
  return result;
}
