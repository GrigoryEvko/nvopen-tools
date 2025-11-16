// Function: sub_382A130
// Address: 0x382a130
//
__int64 *__fastcall sub_382A130(__int64 a1, __int64 a2, int a3, __int64 a4, __int64 a5)
{
  __int64 v5; // r10
  unsigned __int64 v8; // rax
  const __m128i *v9; // rbx
  __int64 v10; // r14
  unsigned __int64 v11; // r14
  const __m128i *v12; // r12
  __m128i *v13; // rax
  int v14; // edx
  _BYTE *v15; // rsi
  __int64 *v16; // r12
  __int64 v18; // [rsp+8h] [rbp-D8h]
  __int64 v19; // [rsp+10h] [rbp-D0h] BYREF
  int v20; // [rsp+18h] [rbp-C8h]
  _BYTE *v21; // [rsp+20h] [rbp-C0h] BYREF
  __int64 v22; // [rsp+28h] [rbp-B8h]
  _BYTE v23[176]; // [rsp+30h] [rbp-B0h] BYREF

  v5 = a1;
  v8 = *(unsigned int *)(a2 + 64);
  v9 = *(const __m128i **)(a2 + 40);
  v19 = 0;
  v10 = 5 * v8;
  v21 = v23;
  v8 *= 40LL;
  v22 = 0x800000000LL;
  v20 = 0;
  v11 = 0xCCCCCCCCCCCCCCCDLL * v10;
  v12 = (const __m128i *)((char *)v9 + v8);
  if ( v8 > 0x140 )
  {
    sub_C8D5F0((__int64)&v21, v23, v11, 0x10u, a5, (__int64)v23);
    v14 = v22;
    v15 = v21;
    v5 = a1;
    v13 = (__m128i *)&v21[16 * (unsigned int)v22];
  }
  else
  {
    v13 = (__m128i *)v23;
    v14 = 0;
    v15 = v23;
  }
  if ( v9 != v12 )
  {
    do
    {
      if ( v13 )
        *v13 = _mm_loadu_si128(v9);
      v9 = (const __m128i *)((char *)v9 + 40);
      ++v13;
    }
    while ( v12 != v9 );
    v15 = v21;
    v14 = v22;
  }
  LODWORD(v22) = v14 + v11;
  v18 = v5;
  sub_375E510(v5, *(_QWORD *)&v15[16 * a3], *(_QWORD *)&v15[16 * a3 + 8], (__int64)&v15[16 * a3], (__int64)&v19);
  v16 = sub_33EC210(*(_QWORD **)(v18 + 8), (__int64 *)a2, (__int64)v21, (unsigned int)v22);
  if ( v21 != v23 )
    _libc_free((unsigned __int64)v21);
  return v16;
}
