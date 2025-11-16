// Function: sub_3840BB0
// Address: 0x3840bb0
//
unsigned __int64 __fastcall sub_3840BB0(
        __int64 a1,
        unsigned __int64 a2,
        unsigned int a3,
        __m128i a4,
        __int64 a5,
        __int64 a6,
        __int64 a7)
{
  __int64 v7; // r15
  int v8; // edx
  unsigned __int64 v9; // r12
  __int64 v10; // rax
  const __m128i *v11; // rbx
  unsigned __int64 v12; // r8
  unsigned __int64 v13; // rcx
  __m128 *v14; // rax
  const __m128i *v15; // r14
  __int64 *v16; // rax
  unsigned __int64 v17; // rsi
  __int64 v18; // rdx
  unsigned __int8 *v19; // rax
  int v20; // edx
  int v21; // esi
  __int64 v22; // rdx
  __int64 *v23; // rax
  unsigned __int64 v24; // r14
  unsigned __int64 v25; // rsi
  __int64 v27; // rax
  _BYTE *v28; // r15
  int v29; // edx
  unsigned __int8 *v30; // rax
  int v31; // edx
  int v32; // esi
  int v33; // [rsp+0h] [rbp-E0h]
  int v34; // [rsp+18h] [rbp-C8h]
  _BYTE *v35; // [rsp+50h] [rbp-90h] BYREF
  __int64 v36; // [rsp+58h] [rbp-88h]
  _BYTE v37[128]; // [rsp+60h] [rbp-80h] BYREF

  v7 = a3;
  v8 = 0;
  v9 = a2;
  v10 = *(unsigned int *)(a2 + 64);
  v11 = *(const __m128i **)(a2 + 40);
  v36 = 0x500000000LL;
  v10 *= 5;
  v12 = 0xCCCCCCCCCCCCCCCDLL * v10;
  v13 = 8 * v10;
  v14 = (__m128 *)v37;
  v15 = (const __m128i *)((char *)v11 + v13);
  v35 = v37;
  if ( v13 > 0xC8 )
  {
    v33 = v12;
    sub_C8D5F0((__int64)&v35, v37, v12, 0x10u, v12, a7);
    v8 = v36;
    LODWORD(v12) = v33;
    v14 = (__m128 *)&v35[16 * (unsigned int)v36];
  }
  if ( v11 != v15 )
  {
    do
    {
      if ( v14 )
      {
        a4 = _mm_loadu_si128(v11);
        *v14 = (__m128)a4;
      }
      v11 = (const __m128i *)((char *)v11 + 40);
      ++v14;
    }
    while ( v15 != v11 );
    v8 = v36;
  }
  LODWORD(v36) = v12 + v8;
  v16 = *(__int64 **)(a2 + 40);
  if ( (_DWORD)v7 == 2 )
  {
    v30 = sub_375B580(
            a1,
            v16[10],
            a4,
            v16[11],
            **(unsigned __int16 **)(a2 + 48),
            *(_QWORD *)(*(_QWORD *)(a2 + 48) + 8LL));
    v32 = v31;
    v22 = (__int64)v35;
    *((_QWORD *)v35 + 4) = v30;
    *(_DWORD *)(v22 + 40) = v32;
  }
  else if ( (_DWORD)v7 == 4 )
  {
    v17 = v16[20];
    v18 = v16[21];
    if ( (*(_WORD *)(v9 + 32) & 0x380) != 0 )
      v19 = sub_37AF270(a1, v17, v18, a4);
    else
      v19 = sub_383B380(a1, v17, v18);
    v21 = v20;
    v22 = (__int64)v35;
    *((_QWORD *)v35 + 8) = v19;
    *(_DWORD *)(v22 + 72) = v21;
  }
  else
  {
    v27 = sub_37AE0F0(a1, v16[5 * v7], v16[5 * v7 + 1]);
    v28 = &v35[16 * v7];
    v34 = v29;
    *(_QWORD *)v28 = v27;
    v22 = (__int64)v35;
    *((_DWORD *)v28 + 2) = v34;
  }
  v23 = sub_33EC210(*(_QWORD **)(a1 + 8), (__int64 *)v9, v22, (unsigned int)v36);
  v24 = (unsigned __int64)v23;
  if ( (__int64 *)v9 != v23 )
  {
    sub_3760E70(a1, v9, 0, (unsigned __int64)v23, 0);
    v25 = v9;
    v9 = 0;
    sub_3760E70(a1, v25, 1, v24, 1);
  }
  if ( v35 != v37 )
    _libc_free((unsigned __int64)v35);
  return v9;
}
