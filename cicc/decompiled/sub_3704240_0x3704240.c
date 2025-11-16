// Function: sub_3704240
// Address: 0x3704240
//
unsigned __int64 *__fastcall sub_3704240(unsigned __int64 *a1, __int64 a2, int a3)
{
  bool v5; // cf
  __int64 v6; // r13
  __int64 v7; // r14
  __m128i *v8; // rax
  unsigned __int64 v9; // rdi
  unsigned __int64 v10; // rcx
  __m128i *v11; // r13
  const __m128i *v12; // rdx
  __m128i *v13; // rcx
  __int64 v14; // rdi
  __int64 v15; // rax
  unsigned int v16; // r13d
  __int64 v17; // r8
  char v18; // r15
  int v19; // r14d
  __m128i *v20; // rsi
  int v21; // edx
  __m128i v22; // rax
  bool v23; // zf
  __int64 v24; // r8
  __int64 v26; // [rsp+20h] [rbp-70h]
  _WORD v28[2]; // [rsp+37h] [rbp-59h] BYREF
  int v29; // [rsp+3Bh] [rbp-55h]
  char v30; // [rsp+3Fh] [rbp-51h]
  _QWORD v31[2]; // [rsp+40h] [rbp-50h] BYREF
  __m128i v32[4]; // [rsp+50h] [rbp-40h] BYREF

  v5 = *(_DWORD *)(a2 + 32) == 0;
  v28[0] = 2;
  v31[1] = 4;
  v28[1] = v5 ? 4611 : 4614;
  v31[0] = v28;
  sub_370CE40(v32, a2 + 144, v31);
  if ( (v32[0].m128i_i64[0] & 0xFFFFFFFFFFFFFFFELL) != 0 )
    BUG();
  *a1 = 0;
  a1[1] = 0;
  a1[2] = 0;
  v6 = *(unsigned int *)(a2 + 8);
  if ( *(_DWORD *)(a2 + 8) )
  {
    v7 = v6;
    v8 = (__m128i *)sub_22077B0(16 * v6);
    v9 = *a1;
    v10 = a1[1];
    v11 = v8;
    v12 = (const __m128i *)*a1;
    if ( v10 != *a1 )
    {
      v13 = (__m128i *)((char *)v8 + v10 - v9);
      do
      {
        if ( v8 )
          *v8 = _mm_loadu_si128(v12);
        ++v8;
        ++v12;
      }
      while ( v13 != v8 );
    }
    if ( v9 )
      j_j___libc_free_0(v9);
    *a1 = (unsigned __int64)v11;
    a1[1] = (unsigned __int64)v11;
    a1[2] = (unsigned __int64)&v11[v7];
    v14 = *(_QWORD *)a2;
    v15 = *(unsigned int *)(a2 + 8);
    v29 = 0;
    v16 = *(_DWORD *)(a2 + 136);
    v17 = v14 + 4 * v15;
    v30 = 0;
    if ( v17 != v14 )
    {
      v18 = 0;
      v19 = a3 + 1;
      do
      {
        v21 = v16;
        v16 = *(_DWORD *)(v17 - 4);
        v26 = v17;
        v30 = v18;
        v22.m128i_i64[0] = (__int64)sub_3703310(
                                      a2,
                                      v16,
                                      v21,
                                      ((unsigned __int64)HIBYTE(v29) << 24)
                                    | (unsigned __int16)v29
                                    | ((unsigned __int64)BYTE2(v29) << 16)
                                    | ((unsigned __int64)(v18 & 1) << 32));
        v20 = (__m128i *)a1[1];
        v23 = v20 == (__m128i *)a1[2];
        v32[0] = v22;
        v24 = v26;
        if ( v23 )
        {
          sub_37040C0(a1, v20, v32);
          v24 = v26;
        }
        else
        {
          if ( v20 )
          {
            *v20 = v22;
            v20 = (__m128i *)a1[1];
          }
          a1[1] = (unsigned __int64)&v20[1];
        }
        v29 = a3;
        a3 = v19;
        if ( !v18 )
          v18 = 1;
        v17 = v24 - 4;
        ++v19;
      }
      while ( v14 != v17 );
    }
  }
  if ( *(_BYTE *)(a2 + 36) )
    *(_BYTE *)(a2 + 36) = 0;
  return a1;
}
