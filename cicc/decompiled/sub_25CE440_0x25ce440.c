// Function: sub_25CE440
// Address: 0x25ce440
//
__int64 __fastcall sub_25CE440(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6, __int64 a7)
{
  __m128i *v7; // r13
  unsigned __int64 *v8; // r14
  unsigned __int64 *v11; // r12
  __int64 v12; // rcx
  __int64 v13; // rdx
  unsigned __int64 v14; // r12
  unsigned __int64 v15; // rax
  unsigned __int64 v16; // rsi
  __int64 v17; // r13
  __int64 v18; // rdx
  __m128i *v19; // r14
  __int64 v20; // rax
  _BYTE *v21; // rsi
  __int64 v22; // rdx
  __m128i *v23; // r14
  __m128i *v24; // rdi
  size_t v25; // rdx
  __int64 v27; // r12
  unsigned __int64 *v28; // r14
  _QWORD v29[2]; // [rsp+10h] [rbp-70h] BYREF
  int v30; // [rsp+20h] [rbp-60h]
  __m128i *v31; // [rsp+30h] [rbp-50h] BYREF
  size_t n; // [rsp+38h] [rbp-48h]
  _QWORD v33[8]; // [rsp+40h] [rbp-40h] BYREF

  if ( *(_WORD *)a1 == 8 )
  {
    v7 = *(__m128i **)a2;
    v8 = *(unsigned __int64 **)(a2 + 8);
    if ( *(unsigned __int64 **)a2 == v8 )
    {
      v12 = *(_QWORD *)(a1 + 16);
      v13 = *(_QWORD *)(a1 + 8);
      v16 = 0xCCCCCCCCCCCCCCCDLL * ((v12 - v13) >> 3);
      if ( !v16 )
        goto LABEL_9;
      v15 = 0;
    }
    else
    {
      v11 = *(unsigned __int64 **)a2;
      do
      {
        if ( (unsigned __int64 *)*v11 != v11 + 2 )
          j_j___libc_free_0(*v11);
        v11 += 4;
      }
      while ( v8 != v11 );
      *(_QWORD *)(a2 + 8) = v7;
      v12 = *(_QWORD *)(a1 + 16);
      v13 = *(_QWORD *)(a1 + 8);
      v14 = 0xCCCCCCCCCCCCCCCDLL * ((v12 - v13) >> 3);
      v15 = ((__int64)v7->m128i_i64 - *(_QWORD *)a2) >> 5;
      v16 = v14;
      if ( v14 <= v15 )
      {
        if ( v14 < v15 )
        {
          v27 = *(_QWORD *)a2 - 0x6666666666666660LL * ((*(_QWORD *)(a1 + 16) - v13) >> 3);
          if ( v7 != (__m128i *)v27 )
          {
            v28 = (unsigned __int64 *)(*(_QWORD *)a2 - 0x6666666666666660LL * ((*(_QWORD *)(a1 + 16) - v13) >> 3));
            do
            {
              if ( (unsigned __int64 *)*v28 != v28 + 2 )
                j_j___libc_free_0(*v28);
              v28 += 4;
            }
            while ( v7 != (__m128i *)v28 );
            *(_QWORD *)(a2 + 8) = v27;
            v12 = *(_QWORD *)(a1 + 16);
            v13 = *(_QWORD *)(a1 + 8);
          }
        }
LABEL_9:
        v17 = 0;
        if ( v12 == v13 )
          return 1;
        while ( 1 )
        {
          v30 = v17;
          v19 = *(__m128i **)a2;
          v20 = v13 + 40 * v17;
          v29[0] = &a7;
          v29[1] = 0;
          if ( *(_WORD *)v20 != 6 && *(_WORD *)v20 != 5 )
          {
            sub_C6AFF0(v29, (__int64)"expected string", 15);
            return 0;
          }
          v21 = *(_BYTE **)(v20 + 8);
          v22 = *(_QWORD *)(v20 + 16);
          v31 = (__m128i *)v33;
          v23 = &v19[2 * v17];
          sub_25CCF50((__int64 *)&v31, v21, (__int64)&v21[v22]);
          v24 = (__m128i *)v23->m128i_i64[0];
          if ( v31 == (__m128i *)v33 )
          {
            v25 = n;
            if ( n )
            {
              if ( n == 1 )
                v24->m128i_i8[0] = v33[0];
              else
                memcpy(v24, v33, n);
              v25 = n;
              v24 = (__m128i *)v23->m128i_i64[0];
            }
            v23->m128i_i64[1] = v25;
            v24->m128i_i8[v25] = 0;
            v24 = v31;
            goto LABEL_14;
          }
          if ( v24 == &v23[1] )
            break;
          v23->m128i_i64[0] = (__int64)v31;
          v18 = v23[1].m128i_i64[0];
          v23->m128i_i64[1] = n;
          v23[1].m128i_i64[0] = v33[0];
          if ( !v24 )
            goto LABEL_29;
          v31 = v24;
          v33[0] = v18;
LABEL_14:
          n = 0;
          v24->m128i_i8[0] = 0;
          if ( v31 != (__m128i *)v33 )
            j_j___libc_free_0((unsigned __int64)v31);
          v13 = *(_QWORD *)(a1 + 8);
          if ( ++v17 >= 0xCCCCCCCCCCCCCCCDLL * ((*(_QWORD *)(a1 + 16) - v13) >> 3) )
            return 1;
        }
        v23->m128i_i64[0] = (__int64)v31;
        v23->m128i_i64[1] = n;
        v23[1].m128i_i64[0] = v33[0];
LABEL_29:
        v31 = (__m128i *)v33;
        v24 = (__m128i *)v33;
        goto LABEL_14;
      }
    }
    sub_1885B60((__m128i **)a2, v16 - v15);
    v12 = *(_QWORD *)(a1 + 16);
    v13 = *(_QWORD *)(a1 + 8);
    goto LABEL_9;
  }
  sub_C6AFF0(&a7, (__int64)"expected array", 14);
  return 0;
}
