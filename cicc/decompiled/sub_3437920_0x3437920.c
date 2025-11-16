// Function: sub_3437920
// Address: 0x3437920
//
__int64 __fastcall sub_3437920(__int64 a1, __int64 a2, __m128i *a3, __int64 a4, __int64 a5, __int64 a6)
{
  unsigned __int64 v9; // rcx
  const __m128i *v10; // r14
  __m128i *v11; // rbx
  __int64 v12; // rdx
  const __m128i *v13; // rax
  __m128i *v15; // rax
  char v16; // dl
  _QWORD *v17; // rdi
  __int64 v18; // rdx
  __int64 v19; // rsi
  __m128i *v20; // rax
  __m128i si128; // xmm0
  const __m128i *v22; // rdx
  __int64 v23; // rax
  __m128i v24[4]; // [rsp+10h] [rbp-40h] BYREF

  if ( *(_QWORD *)(a2 + 184) )
  {
    v15 = sub_3437040(a2 + 144, a3);
    *(_BYTE *)(a1 + 8) = 0;
    *(_QWORD *)a1 = v15;
    *(_BYTE *)(a1 + 16) = v16;
  }
  else
  {
    v9 = *(unsigned int *)(a2 + 8);
    v10 = *(const __m128i **)a2;
    v11 = (__m128i *)(*(_QWORD *)a2 + 16 * v9);
    if ( *(__m128i **)a2 == v11 )
    {
      if ( v9 > 7 )
      {
        v17 = (_QWORD *)(a2 + 144);
LABEL_16:
        *(_DWORD *)(a2 + 8) = 0;
        v20 = sub_3437040((__int64)v17, a3);
        *(_BYTE *)(a1 + 8) = 0;
        *(_QWORD *)a1 = v20;
        *(_BYTE *)(a1 + 16) = 1;
        return a1;
      }
    }
    else
    {
      v12 = a3->m128i_i64[0];
      v13 = *(const __m128i **)a2;
      while ( v13->m128i_i64[0] != v12 || v13->m128i_i32[2] != a3->m128i_i32[2] )
      {
        if ( v11 == ++v13 )
          goto LABEL_11;
      }
      if ( v13 != v11 )
      {
        *(_BYTE *)(a1 + 8) = 1;
        *(_QWORD *)a1 = v13;
        *(_BYTE *)(a1 + 16) = 0;
        return a1;
      }
LABEL_11:
      if ( v9 > 7 )
      {
        v17 = (_QWORD *)(a2 + 144);
        do
        {
          v24[0].m128i_i64[0] = (__int64)v17;
          v19 = sub_3055200(v17, a2 + 152, (__int64)v10);
          if ( v18 )
          {
            sub_3433AF0((__int64)v17, v19, v18, v10);
            v17 = (_QWORD *)v24[0].m128i_i64[0];
          }
          ++v10;
        }
        while ( v11 != v10 );
        goto LABEL_16;
      }
    }
    si128 = _mm_loadu_si128(a3);
    if ( v9 + 1 > *(unsigned int *)(a2 + 12) )
    {
      v24[0] = si128;
      sub_C8D5F0(a2, (const void *)(a2 + 16), v9 + 1, 0x10u, a5, a6);
      si128 = _mm_load_si128(v24);
      v11 = (__m128i *)(*(_QWORD *)a2 + 16LL * *(unsigned int *)(a2 + 8));
    }
    *v11 = si128;
    v22 = *(const __m128i **)a2;
    v23 = (unsigned int)(*(_DWORD *)(a2 + 8) + 1);
    *(_DWORD *)(a2 + 8) = v23;
    *(_BYTE *)(a1 + 8) = 1;
    *(_QWORD *)a1 = &v22[v23 - 1];
    *(_BYTE *)(a1 + 16) = 1;
  }
  return a1;
}
