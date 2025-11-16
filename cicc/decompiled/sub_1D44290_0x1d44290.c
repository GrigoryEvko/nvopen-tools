// Function: sub_1D44290
// Address: 0x1d44290
//
__int64 *__fastcall sub_1D44290(
        __int64 *a1,
        __int64 a2,
        __int64 a3,
        unsigned __int64 a4,
        const void **a5,
        __m128 a6,
        double a7,
        __m128i a8,
        __int64 a9,
        __int128 a10)
{
  unsigned int v10; // r10d
  unsigned int v12; // r13d
  const __m128i *v13; // r9
  __int64 *result; // rax
  const __m128i *v15; // rbx
  __int64 v16; // rax
  const __m128i *v17; // r11
  unsigned __int64 v18; // rdx
  __m128 *v19; // rax
  int v20; // esi
  _BYTE *v21; // rcx
  __int64 v22; // rdx
  __int128 v23; // [rsp-10h] [rbp-100h]
  unsigned int v24; // [rsp+8h] [rbp-E8h]
  const void **v25; // [rsp+10h] [rbp-E0h]
  const __m128i *v26; // [rsp+18h] [rbp-D8h]
  __int64 *v27; // [rsp+20h] [rbp-D0h]
  unsigned __int64 v28; // [rsp+20h] [rbp-D0h]
  _BYTE *v29; // [rsp+30h] [rbp-C0h] BYREF
  __int64 v30; // [rsp+38h] [rbp-B8h]
  _BYTE v31[176]; // [rsp+40h] [rbp-B0h] BYREF
  __int128 v32; // [rsp+100h] [rbp+10h]

  v10 = a4;
  v12 = a2;
  v13 = (const __m128i *)a10;
  if ( *((_QWORD *)&a10 + 1) == 2 )
    return sub_1D332F0(
             a1,
             a2,
             a3,
             a4,
             a5,
             0,
             *(double *)a6.m128_u64,
             a7,
             a8,
             *(_QWORD *)a10,
             *(_QWORD *)(a10 + 8),
             *(_OWORD *)(a10 + 40));
  if ( *((_QWORD *)&a10 + 1) > 2u )
  {
    if ( *((_QWORD *)&a10 + 1) == 3 )
    {
      return sub_1D3A900(
               a1,
               a2,
               a3,
               a4,
               a5,
               0,
               a6,
               a7,
               a8,
               *(_QWORD *)a10,
               *(__int16 **)(a10 + 8),
               *(_OWORD *)(a10 + 40),
               *(_QWORD *)(a10 + 80),
               *(_QWORD *)(a10 + 88));
    }
    else
    {
      v15 = (const __m128i *)a10;
      v16 = 40LL * *((_QWORD *)&a10 + 1);
      v30 = 0x800000000LL;
      v29 = v31;
      v17 = (const __m128i *)(a10 + 40LL * *((_QWORD *)&a10 + 1));
      v18 = 0xCCCCCCCCCCCCCCCDLL * ((40LL * *((_QWORD *)&a10 + 1)) >> 3);
      if ( (unsigned __int64)(40LL * *((_QWORD *)&a10 + 1)) > 0x140 )
      {
        v24 = a4;
        v25 = a5;
        v26 = (const __m128i *)(a10 + v16);
        v28 = 0xCCCCCCCCCCCCCCCDLL * (v16 >> 3);
        sub_16CD150((__int64)&v29, v31, v18, 16, (int)a5, a10);
        v20 = v30;
        v21 = v29;
        LODWORD(v18) = v28;
        v17 = v26;
        a5 = v25;
        v10 = v24;
        v13 = (const __m128i *)a10;
        v19 = (__m128 *)&v29[16 * (unsigned int)v30];
      }
      else
      {
        v19 = (__m128 *)v31;
        v20 = 0;
        v21 = v31;
      }
      if ( v13 != v17 )
      {
        do
        {
          if ( v19 )
          {
            a6 = (__m128)_mm_loadu_si128(v15);
            *v19 = a6;
          }
          v15 = (const __m128i *)((char *)v15 + 40);
          ++v19;
        }
        while ( v17 != v15 );
        v21 = v29;
        v20 = v30;
      }
      v22 = (unsigned int)(v18 + v20);
      LODWORD(v30) = v22;
      *((_QWORD *)&v23 + 1) = v22;
      *(_QWORD *)&v23 = v21;
      result = sub_1D359D0(a1, v12, a3, v10, a5, 0, *(double *)a6.m128_u64, a7, a8, v23);
      if ( v29 != v31 )
      {
        v27 = result;
        _libc_free((unsigned __int64)v29);
        return v27;
      }
    }
  }
  else if ( *((_QWORD *)&a10 + 1) )
  {
    v32 = (__int128)_mm_loadu_si128((const __m128i *)a10);
    return (__int64 *)sub_1D309E0(
                        a1,
                        a2,
                        a3,
                        a4,
                        a5,
                        0,
                        *(double *)a6.m128_u64,
                        *(double *)&v32,
                        *(double *)a8.m128i_i64,
                        v32);
  }
  else
  {
    return sub_1D2B300(a1, a2, a3, a4, (__int64)a5, a10);
  }
  return result;
}
