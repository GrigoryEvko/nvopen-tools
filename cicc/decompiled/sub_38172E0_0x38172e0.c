// Function: sub_38172E0
// Address: 0x38172e0
//
__int64 *__fastcall sub_38172E0(__int64 a1, __int64 a2, unsigned int a3, __m128i a4)
{
  __int64 v4; // r14
  __int64 *v5; // r12
  unsigned __int8 *v6; // rax
  const __m128i *v7; // rbx
  __int64 v8; // r8
  unsigned __int64 v9; // rax
  int v10; // edx
  int v11; // r10d
  __int64 v12; // rdx
  unsigned __int64 v13; // rdx
  const __m128i *v14; // r13
  __m128i *v15; // rax
  int v16; // esi
  _BYTE *v17; // rcx
  _BYTE *v18; // rcx
  __int64 v19; // rdx
  __int64 *v20; // rax
  unsigned __int64 v21; // r13
  unsigned __int64 v22; // rsi
  int v24; // [rsp+0h] [rbp-B0h]
  __int64 v25; // [rsp+8h] [rbp-A8h]
  int v26; // [rsp+10h] [rbp-A0h]
  _BYTE *v27; // [rsp+30h] [rbp-80h] BYREF
  __int64 v28; // [rsp+38h] [rbp-78h]
  _BYTE v29[112]; // [rsp+40h] [rbp-70h] BYREF

  v4 = a3;
  v5 = (__int64 *)a2;
  v6 = sub_375B580(
         a1,
         *(_QWORD *)(*(_QWORD *)(a2 + 40) + 40LL * a3),
         a4,
         *(_QWORD *)(*(_QWORD *)(a2 + 40) + 40LL * a3 + 8),
         **(unsigned __int16 **)(a2 + 48),
         *(_QWORD *)(*(_QWORD *)(a2 + 48) + 8LL));
  v7 = *(const __m128i **)(a2 + 40);
  v8 = (__int64)v6;
  v9 = *(unsigned int *)(a2 + 64);
  v11 = v10;
  v27 = v29;
  v12 = 5 * v9;
  v28 = 0x400000000LL;
  v9 *= 40LL;
  v13 = 0xCCCCCCCCCCCCCCCDLL * v12;
  v14 = (const __m128i *)((char *)v7 + v9);
  if ( v9 > 0xA0 )
  {
    v24 = v11;
    v25 = v8;
    v26 = v13;
    sub_C8D5F0((__int64)&v27, v29, v13, 0x10u, v8, (__int64)v29);
    v16 = v28;
    v17 = v27;
    LODWORD(v13) = v26;
    v8 = v25;
    v11 = v24;
    v15 = (__m128i *)&v27[16 * (unsigned int)v28];
  }
  else
  {
    v15 = (__m128i *)v29;
    v16 = 0;
    v17 = v29;
  }
  if ( v7 != v14 )
  {
    do
    {
      if ( v15 )
        *v15 = _mm_loadu_si128(v7);
      v7 = (const __m128i *)((char *)v7 + 40);
      ++v15;
    }
    while ( v14 != v7 );
    v17 = v27;
    v16 = v28;
  }
  v18 = &v17[16 * v4];
  LODWORD(v28) = v16 + v13;
  *(_QWORD *)v18 = v8;
  v19 = (__int64)v27;
  *((_DWORD *)v18 + 2) = v11;
  v20 = sub_33EC210(*(_QWORD **)(a1 + 8), v5, v19, (unsigned int)v28);
  v21 = (unsigned __int64)v20;
  if ( v5 != v20 )
  {
    sub_3760E70(a1, (unsigned __int64)v5, 0, (unsigned __int64)v20, 0);
    v22 = (unsigned __int64)v5;
    v5 = 0;
    sub_3760E70(a1, v22, 1, v21, 1);
  }
  if ( v27 != v29 )
    _libc_free((unsigned __int64)v27);
  return v5;
}
