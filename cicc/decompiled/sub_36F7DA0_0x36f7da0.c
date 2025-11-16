// Function: sub_36F7DA0
// Address: 0x36f7da0
//
void __fastcall sub_36F7DA0(__int64 a1, _BYTE *a2, size_t a3)
{
  __int64 v5; // rbx
  __int64 v6; // r14
  _QWORD *v7; // rax
  __int64 v8; // rax
  __m128i *v9; // rsi
  unsigned __int64 v10; // rdx
  __m128i *v11; // rax
  __int64 v12; // rcx
  _QWORD *v13; // rdi
  _QWORD *v14; // rdi
  __int64 v15; // rdi
  char *v16; // r12
  unsigned __int64 v17; // [rsp+0h] [rbp-70h]
  unsigned int v18; // [rsp+8h] [rbp-68h]
  size_t v19; // [rsp+18h] [rbp-58h] BYREF
  _QWORD *v20; // [rsp+20h] [rbp-50h] BYREF
  unsigned __int64 v21; // [rsp+28h] [rbp-48h]
  _QWORD v22[8]; // [rsp+30h] [rbp-40h] BYREF

  v17 = a3;
  v18 = *(_DWORD *)(a1 + 16);
  if ( !v18 )
  {
LABEL_8:
    if ( !a2 )
    {
      v21 = 0;
      v20 = v22;
      LOBYTE(v22[0]) = 0;
      goto LABEL_14;
    }
    v19 = a3;
    v20 = v22;
    if ( a3 > 0xF )
    {
      v20 = (_QWORD *)sub_22409D0((__int64)&v20, &v19, 0);
      v14 = v20;
      v22[0] = v19;
    }
    else
    {
      if ( a3 == 1 )
      {
        LOBYTE(v22[0]) = *a2;
        v7 = v22;
LABEL_12:
        v21 = v17;
        *((_BYTE *)v7 + v17) = 0;
        v18 = *(_DWORD *)(a1 + 16);
LABEL_14:
        v8 = v18;
        v9 = (__m128i *)&v20;
        v10 = *(_QWORD *)(a1 + 8);
        if ( (unsigned __int64)v18 + 1 > *(unsigned int *)(a1 + 20) )
        {
          v15 = a1 + 8;
          if ( v10 > (unsigned __int64)&v20 || (unsigned __int64)&v20 >= v10 + 32LL * v18 )
          {
            sub_95D880(v15, v18 + 1LL);
            v8 = *(unsigned int *)(a1 + 16);
            v10 = *(_QWORD *)(a1 + 8);
            v18 = *(_DWORD *)(a1 + 16);
            v9 = (__m128i *)&v20;
          }
          else
          {
            v16 = (char *)&v20 - v10;
            sub_95D880(v15, v18 + 1LL);
            v10 = *(_QWORD *)(a1 + 8);
            v8 = *(unsigned int *)(a1 + 16);
            v9 = (__m128i *)&v16[v10];
            v18 = *(_DWORD *)(a1 + 16);
          }
        }
        v11 = (__m128i *)(v10 + 32 * v8);
        if ( v11 )
        {
          v11->m128i_i64[0] = (__int64)v11[1].m128i_i64;
          if ( (__m128i *)v9->m128i_i64[0] == &v9[1] )
          {
            v11[1] = _mm_loadu_si128(v9 + 1);
          }
          else
          {
            v11->m128i_i64[0] = v9->m128i_i64[0];
            v11[1].m128i_i64[0] = v9[1].m128i_i64[0];
          }
          v12 = v9->m128i_i64[1];
          v9->m128i_i64[0] = (__int64)v9[1].m128i_i64;
          v9->m128i_i64[1] = 0;
          v11->m128i_i64[1] = v12;
          v9[1].m128i_i8[0] = 0;
          v18 = *(_DWORD *)(a1 + 16);
        }
        v13 = v20;
        *(_DWORD *)(a1 + 16) = v18 + 1;
        if ( v13 != v22 )
          j_j___libc_free_0((unsigned __int64)v13);
        return;
      }
      if ( !a3 )
      {
        v7 = v22;
        goto LABEL_12;
      }
      v14 = v22;
    }
    memcpy(v14, a2, a3);
    v17 = v19;
    v7 = v20;
    goto LABEL_12;
  }
  v5 = *(_QWORD *)(a1 + 8);
  v6 = v5 + 32LL * (v18 - 1) + 32;
  while ( a3 != *(_QWORD *)(v5 + 8) || a3 && memcmp(*(const void **)v5, a2, a3) )
  {
    v5 += 32;
    if ( v6 == v5 )
      goto LABEL_8;
  }
}
