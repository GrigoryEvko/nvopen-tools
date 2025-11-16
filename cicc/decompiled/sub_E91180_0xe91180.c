// Function: sub_E91180
// Address: 0xe91180
//
void __fastcall sub_E91180(_QWORD *a1, _QWORD *a2, unsigned int **a3)
{
  unsigned int *v5; // r12
  unsigned int *v6; // r15
  __int64 v7; // rdx
  __int64 *v8; // r14
  unsigned int *v9; // rax
  unsigned int *v10; // r12
  __int64 v11; // r14
  unsigned __int64 v12; // rax
  unsigned int *v13; // r14
  __int64 v14; // rsi
  unsigned int *v15; // [rsp+8h] [rbp-68h]
  __int64 v16; // [rsp+18h] [rbp-58h] BYREF
  __int64 v17; // [rsp+20h] [rbp-50h] BYREF
  unsigned int *v18; // [rsp+28h] [rbp-48h]
  unsigned int *v19; // [rsp+30h] [rbp-40h]

  (*(void (__fastcall **)(_QWORD *, _QWORD, __int64))(*a2 + 536LL))(a2, a1[10], 8);
  if ( *(_QWORD *)(a1[11] + 80LL) || a1[10] == *((_QWORD *)*a3 + 2) )
  {
    sub_E98EB0(a2, (__int64)(a1[8] - a1[7]) >> 5, 0);
    sub_E98EB0(a2, a1[3], 0);
  }
  else
  {
    sub_E98EB0(a2, ((__int64)(a1[8] - a1[7]) >> 5) + 1, 0);
    sub_E98EB0(a2, a1[3], 0);
    sub_E90B40(*a3, a2, 0);
  }
  v15 = (unsigned int *)a1[8];
  if ( v15 != (unsigned int *)a1[7] )
  {
    v5 = *a3;
    v6 = (unsigned int *)a1[7];
    do
    {
      v7 = (__int64)v5;
      v5 = v6;
      sub_E90B40(v6, a2, v7);
      *a3 = v6;
      v6 += 8;
    }
    while ( v15 != v6 );
  }
  v8 = (__int64 *)a1[2];
  v9 = 0;
  v10 = 0;
  v17 = 0;
  v18 = 0;
  v19 = 0;
  if ( v8 )
  {
    while ( 1 )
    {
      v16 = v8[3];
      if ( v9 == v10 )
      {
        sub_E90980((unsigned __int64 *)&v17, (char *)v10, (const __m128i *)(v8 + 1), (unsigned __int64 *)&v16);
        v8 = (__int64 *)*v8;
        v10 = v18;
        if ( !v8 )
          goto LABEL_16;
      }
      else
      {
        if ( v10 )
        {
          *(__m128i *)v10 = _mm_loadu_si128((const __m128i *)(v8 + 1));
          *((_QWORD *)v10 + 2) = v16;
          v10 = v18;
        }
        v10 += 6;
        v18 = v10;
        v8 = (__int64 *)*v8;
        if ( !v8 )
        {
LABEL_16:
          v11 = v17;
          if ( (unsigned int *)v17 != v10 )
          {
            _BitScanReverse64(&v12, 0xAAAAAAAAAAAAAAABLL * (((__int64)v10 - v17) >> 3));
            sub_E90E90(v17, (unsigned __int64)v10, 2LL * (int)(63 - (v12 ^ 0x3F)));
            sub_E8F3B0(v11, (__int64)v10);
            v13 = v18;
            v10 = (unsigned int *)v17;
            if ( (unsigned int *)v17 != v18 )
            {
              do
              {
                v14 = *v10;
                v10 += 6;
                sub_E98EB0(a2, v14, 0);
                sub_E91180(*((_QWORD *)v10 - 1), a2, a3);
              }
              while ( v13 != v10 );
              v10 = (unsigned int *)v17;
            }
          }
          if ( v10 )
            j_j___libc_free_0(v10, (char *)v19 - (char *)v10);
          return;
        }
      }
      v9 = v19;
    }
  }
}
