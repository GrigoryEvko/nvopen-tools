// Function: sub_21293C0
// Address: 0x21293c0
//
__int64 *__fastcall sub_21293C0(__int64 a1, __int64 a2, unsigned int a3, double a4, double a5, double a6)
{
  __int64 v6; // r14
  __int64 v8; // rax
  const __m128i *v9; // rbx
  __int64 v10; // r9
  __int64 v11; // rax
  int v12; // edx
  int v13; // r10d
  __int64 v14; // rax
  const __m128i *v15; // r12
  unsigned __int64 v16; // rdx
  __m128i *v17; // rax
  int v18; // esi
  _BYTE *v19; // rcx
  _BYTE *v20; // rcx
  __int64 v21; // rdx
  __int64 *v22; // r12
  int v24; // [rsp+0h] [rbp-B0h]
  __int64 v25; // [rsp+8h] [rbp-A8h]
  unsigned __int64 v26; // [rsp+10h] [rbp-A0h]
  _BYTE *v27; // [rsp+30h] [rbp-80h] BYREF
  __int64 v28; // [rsp+38h] [rbp-78h]
  _BYTE v29[112]; // [rsp+40h] [rbp-70h] BYREF

  v6 = a3;
  v8 = sub_200E230(
         (_QWORD *)a1,
         *(_QWORD *)(*(_QWORD *)(a2 + 32) + 40LL * a3),
         *(_QWORD *)(*(_QWORD *)(a2 + 32) + 40LL * a3 + 8),
         **(unsigned __int8 **)(a2 + 40),
         *(_QWORD *)(*(_QWORD *)(a2 + 40) + 8LL),
         a4,
         a5,
         a6);
  v9 = *(const __m128i **)(a2 + 32);
  v10 = v8;
  v11 = *(unsigned int *)(a2 + 56);
  v13 = v12;
  v27 = v29;
  v28 = 0x400000000LL;
  v14 = 40 * v11;
  v15 = (const __m128i *)((char *)v9 + v14);
  v16 = 0xCCCCCCCCCCCCCCCDLL * (v14 >> 3);
  if ( (unsigned __int64)v14 > 0xA0 )
  {
    v24 = v13;
    v25 = v10;
    v26 = 0xCCCCCCCCCCCCCCCDLL * (v14 >> 3);
    sub_16CD150((__int64)&v27, v29, v16, 16, (int)v29, v10);
    v18 = v28;
    v19 = v27;
    LODWORD(v16) = v26;
    v10 = v25;
    v13 = v24;
    v17 = (__m128i *)&v27[16 * (unsigned int)v28];
  }
  else
  {
    v17 = (__m128i *)v29;
    v18 = 0;
    v19 = v29;
  }
  if ( v9 != v15 )
  {
    do
    {
      if ( v17 )
        *v17 = _mm_loadu_si128(v9);
      v9 = (const __m128i *)((char *)v9 + 40);
      ++v17;
    }
    while ( v15 != v9 );
    v19 = v27;
    v18 = v28;
  }
  v20 = &v19[16 * v6];
  LODWORD(v28) = v18 + v16;
  *(_QWORD *)v20 = v10;
  v21 = (__int64)v27;
  *((_DWORD *)v20 + 2) = v13;
  v22 = sub_1D2E160(*(_QWORD **)(a1 + 8), (__int64 *)a2, v21, (unsigned int)v28);
  if ( v27 != v29 )
    _libc_free((unsigned __int64)v27);
  return v22;
}
