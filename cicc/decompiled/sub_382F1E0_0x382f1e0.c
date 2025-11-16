// Function: sub_382F1E0
// Address: 0x382f1e0
//
__m128i *__fastcall sub_382F1E0(__int64 a1, __int64 a2, int a3, __m128i a4)
{
  __int64 v6; // rax
  __int64 v7; // rbx
  __int64 v8; // r14
  __int64 v9; // r15
  unsigned int v10; // edx
  unsigned int v11; // edi
  __int64 v12; // r8
  __int64 *v13; // r13
  __int64 v14; // rsi
  unsigned __int64 v15; // r9
  __int64 v16; // r10
  __int64 v17; // r11
  bool v18; // al
  unsigned __int64 *v19; // rax
  __int64 *v20; // rcx
  __m128i *v21; // r12
  unsigned __int16 *v23; // rdx
  unsigned __int8 *v24; // r14
  const __m128i *v25; // rbx
  int v26; // edx
  int v27; // r15d
  unsigned __int64 v28; // rdx
  __int64 v29; // r9
  __m128i *v30; // rax
  int v31; // ecx
  _BYTE *v32; // r10
  unsigned int v33; // ecx
  __int128 v34; // [rsp-40h] [rbp-140h]
  __int64 v35; // [rsp+10h] [rbp-F0h]
  __int64 v36; // [rsp+18h] [rbp-E8h]
  __int64 *v37; // [rsp+28h] [rbp-D8h]
  char v38; // [rsp+34h] [rbp-CCh]
  int v39; // [rsp+38h] [rbp-C8h]
  __int8 *v40; // [rsp+38h] [rbp-C8h]
  const __m128i *v41; // [rsp+40h] [rbp-C0h]
  int v42; // [rsp+40h] [rbp-C0h]
  __int128 *v43; // [rsp+48h] [rbp-B8h]
  __int64 v44; // [rsp+50h] [rbp-B0h]
  _BYTE *v45; // [rsp+80h] [rbp-80h] BYREF
  __int64 v46; // [rsp+88h] [rbp-78h]
  _BYTE v47[112]; // [rsp+90h] [rbp-70h] BYREF

  v6 = *(_QWORD *)(a2 + 40);
  v7 = *(_QWORD *)(v6 + 48);
  v8 = *(_QWORD *)(v6 + 160);
  v9 = *(_QWORD *)(v6 + 168);
  if ( a3 == 4 )
  {
    v23 = (unsigned __int16 *)(*(_QWORD *)(*(_QWORD *)(v6 + 40) + 48LL) + 16LL * *(unsigned int *)(v6 + 48));
    v24 = sub_375B580(a1, *(_QWORD *)(v6 + 160), a4, *(_QWORD *)(v6 + 168), *v23, *((_QWORD *)v23 + 1));
    v25 = *(const __m128i **)(a2 + 40);
    v45 = v47;
    v27 = v26;
    v28 = *(unsigned int *)(a2 + 64);
    v46 = 0x400000000LL;
    v29 = (__int64)&v25->m128i_i64[5 * v28];
    if ( 40 * v28 > 0xA0 )
    {
      v40 = &v25->m128i_i8[40 * v28];
      v42 = v28;
      sub_C8D5F0((__int64)&v45, v47, v28, 0x10u, (__int64)v47, v29);
      v31 = v46;
      v32 = v45;
      LODWORD(v28) = v42;
      v29 = (__int64)v40;
      v30 = (__m128i *)&v45[16 * (unsigned int)v46];
    }
    else
    {
      v30 = (__m128i *)v47;
      v31 = 0;
      v32 = v47;
    }
    if ( v25 != (const __m128i *)v29 )
    {
      do
      {
        if ( v30 )
          *v30 = _mm_loadu_si128(v25);
        v25 = (const __m128i *)((char *)v25 + 40);
        ++v30;
      }
      while ( (const __m128i *)v29 != v25 );
      v32 = v45;
      v31 = v46;
    }
    v33 = v28 + v31;
    LODWORD(v46) = v33;
    *((_QWORD *)v32 + 8) = v24;
    *((_DWORD *)v32 + 18) = v27;
    v21 = (__m128i *)sub_33EC210(*(_QWORD **)(a1 + 8), (__int64 *)a2, (__int64)v32, v33);
    if ( v45 != v47 )
      _libc_free((unsigned __int64)v45);
  }
  else
  {
    v44 = sub_37AE0F0(a1, *(_QWORD *)(v6 + 40), *(_QWORD *)(v6 + 48));
    v11 = v10;
    v12 = v44;
    v13 = *(__int64 **)(a1 + 8);
    v14 = *(_QWORD *)(a2 + 80);
    v15 = v7 & 0xFFFFFFFF00000000LL | v10;
    v16 = *(unsigned __int16 *)(a2 + 96);
    v17 = *(_QWORD *)(a2 + 104);
    v18 = (*(_BYTE *)(a2 + 33) & 8) != 0;
    v45 = (_BYTE *)v14;
    v38 = v18;
    v39 = (*(_WORD *)(a2 + 32) >> 7) & 7;
    v41 = *(const __m128i **)(a2 + 112);
    v19 = *(unsigned __int64 **)(a2 + 40);
    v43 = (__int128 *)(v19 + 15);
    v20 = (__int64 *)(v19 + 10);
    if ( v14 )
    {
      v35 = v16;
      v36 = v17;
      v37 = (__int64 *)(v19 + 10);
      sub_B96E90((__int64)&v45, v14, 1);
      v19 = *(unsigned __int64 **)(a2 + 40);
      v12 = v44;
      v15 = v7 & 0xFFFFFFFF00000000LL | v11;
      v16 = v35;
      v17 = v36;
      v20 = v37;
    }
    LODWORD(v46) = *(_DWORD *)(a2 + 72);
    *((_QWORD *)&v34 + 1) = v9;
    *(_QWORD *)&v34 = v8;
    v21 = sub_33F65D0(v13, *v19, v19[1], (__int64)&v45, v12, v15, *v20, v20[1], *v43, v34, v16, v17, v41, v39, 1, v38);
    if ( v45 )
      sub_B91220((__int64)&v45, (__int64)v45);
  }
  return v21;
}
