// Function: sub_38C71D0
// Address: 0x38c71d0
//
void __fastcall sub_38C71D0(__int64 a1, int a2)
{
  __int64 v3; // rbx
  unsigned __int64 v4; // r13
  unsigned __int64 v5; // rdi
  __int64 v6; // rax
  __int64 v7; // rdx
  const __m128i *v8; // rbx
  __int64 i; // rdx
  const __m128i *v10; // r15
  __m128i *v11; // rax
  __int64 v12; // rax
  __int64 v13; // rax
  __int64 v14; // rax
  __int64 v15; // rdx
  __int64 j; // rdx
  __m128i *v17; // [rsp+8h] [rbp-78h] BYREF
  __int64 v18; // [rsp+20h] [rbp-60h]
  __int64 v19; // [rsp+40h] [rbp-40h]

  v3 = *(unsigned int *)(a1 + 24);
  v4 = *(_QWORD *)(a1 + 8);
  v5 = ((((((((((unsigned int)(a2 - 1) | ((unsigned __int64)(unsigned int)(a2 - 1) >> 1)) >> 2)
            | (unsigned int)(a2 - 1)
            | ((unsigned __int64)(unsigned int)(a2 - 1) >> 1)) >> 4)
          | (((unsigned int)(a2 - 1) | ((unsigned __int64)(unsigned int)(a2 - 1) >> 1)) >> 2)
          | (unsigned int)(a2 - 1)
          | ((unsigned __int64)(unsigned int)(a2 - 1) >> 1)) >> 8)
        | (((((unsigned int)(a2 - 1) | ((unsigned __int64)(unsigned int)(a2 - 1) >> 1)) >> 2)
          | (unsigned int)(a2 - 1)
          | ((unsigned __int64)(unsigned int)(a2 - 1) >> 1)) >> 4)
        | (((unsigned int)(a2 - 1) | ((unsigned __int64)(unsigned int)(a2 - 1) >> 1)) >> 2)
        | (unsigned int)(a2 - 1)
        | ((unsigned __int64)(unsigned int)(a2 - 1) >> 1)) >> 16)
      | (((((((unsigned int)(a2 - 1) | ((unsigned __int64)(unsigned int)(a2 - 1) >> 1)) >> 2)
          | (unsigned int)(a2 - 1)
          | ((unsigned __int64)(unsigned int)(a2 - 1) >> 1)) >> 4)
        | (((unsigned int)(a2 - 1) | ((unsigned __int64)(unsigned int)(a2 - 1) >> 1)) >> 2)
        | (unsigned int)(a2 - 1)
        | ((unsigned __int64)(unsigned int)(a2 - 1) >> 1)) >> 8)
      | (((((unsigned int)(a2 - 1) | ((unsigned __int64)(unsigned int)(a2 - 1) >> 1)) >> 2)
        | (unsigned int)(a2 - 1)
        | ((unsigned __int64)(unsigned int)(a2 - 1) >> 1)) >> 4)
      | (((unsigned int)(a2 - 1) | ((unsigned __int64)(unsigned int)(a2 - 1) >> 1)) >> 2)
      | (unsigned int)(a2 - 1)
      | ((unsigned __int64)(unsigned int)(a2 - 1) >> 1))
     + 1;
  if ( (unsigned int)v5 < 0x40 )
    LODWORD(v5) = 64;
  *(_DWORD *)(a1 + 24) = v5;
  v6 = sub_22077B0(32LL * (unsigned int)v5);
  *(_QWORD *)(a1 + 8) = v6;
  if ( v4 )
  {
    v7 = *(unsigned int *)(a1 + 24);
    *(_QWORD *)(a1 + 16) = 0;
    v8 = (const __m128i *)(v4 + 32 * v3);
    for ( i = v6 + 32 * v7; i != v6; v6 += 32 )
    {
      if ( v6 )
      {
        *(_QWORD *)v6 = 0;
        *(_DWORD *)(v6 + 8) = 0;
        *(_DWORD *)(v6 + 12) = -1;
        *(_BYTE *)(v6 + 16) = 0;
        *(_BYTE *)(v6 + 17) = 0;
        *(_DWORD *)(v6 + 20) = 0x7FFFFFFF;
      }
    }
    if ( v8 != (const __m128i *)v4 )
    {
      v10 = (const __m128i *)v4;
      while ( 1 )
      {
        if ( v10->m128i_i64[0] )
          goto LABEL_10;
        v12 = v10->m128i_i64[1];
        if ( v12 == 0xFFFFFFFF00000000LL )
        {
          HIDWORD(v18) = 0x7FFFFFFF;
          v14 = v10[1].m128i_i64[0];
          LOWORD(v18) = 0;
          if ( ((v18 ^ v14) & 0xFFFFFFFF0000FFFFLL) == 0 )
            goto LABEL_11;
          goto LABEL_10;
        }
        if ( v12 != 0xFFFFFFFFLL
          || (v13 = v10[1].m128i_i64[0],
              HIDWORD(v19) = 0x7FFFFFFF,
              LOWORD(v19) = 0,
              ((v19 ^ v13) & 0xFFFFFFFF0000FFFFLL) != 0) )
        {
LABEL_10:
          sub_38C7080(a1, (__int64)v10, &v17);
          v11 = v17;
          *v17 = _mm_loadu_si128(v10);
          v11[1].m128i_i64[0] = v10[1].m128i_i64[0];
          v11[1].m128i_i64[1] = v10[1].m128i_i64[1];
          ++*(_DWORD *)(a1 + 16);
LABEL_11:
          v10 += 2;
          if ( v8 == v10 )
            break;
        }
        else
        {
          v10 += 2;
          if ( v8 == v10 )
            break;
        }
      }
    }
    j___libc_free_0(v4);
  }
  else
  {
    v15 = *(unsigned int *)(a1 + 24);
    *(_QWORD *)(a1 + 16) = 0;
    for ( j = v6 + 32 * v15; j != v6; v6 += 32 )
    {
      if ( v6 )
      {
        *(_QWORD *)v6 = 0;
        *(_DWORD *)(v6 + 8) = 0;
        *(_DWORD *)(v6 + 12) = -1;
        *(_BYTE *)(v6 + 16) = 0;
        *(_BYTE *)(v6 + 17) = 0;
        *(_DWORD *)(v6 + 20) = 0x7FFFFFFF;
      }
    }
  }
}
