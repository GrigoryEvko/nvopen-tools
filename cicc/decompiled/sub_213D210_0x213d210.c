// Function: sub_213D210
// Address: 0x213d210
//
__int64 __fastcall sub_213D210(__int64 a1, __int64 a2, int a3, double a4, double a5, __m128i a6)
{
  _QWORD *v8; // rdx
  __int64 v9; // rsi
  unsigned __int64 v10; // r11
  __int64 v11; // rbx
  __int64 v12; // r14
  __int64 v13; // rax
  __int64 v14; // r15
  unsigned __int8 v15; // r10
  __int64 v16; // r8
  __int64 v17; // rsi
  __int64 v18; // rcx
  __int64 v19; // r12
  __int64 v20; // r14
  const __m128i *v21; // rbx
  int v22; // edx
  int v23; // r15d
  __int64 v24; // rax
  const __m128i *v25; // r9
  unsigned __int64 v26; // rdx
  __m128i *v27; // rax
  int v28; // ecx
  _BYTE *v29; // r10
  unsigned int v30; // ecx
  unsigned int v31; // edx
  __int128 v33; // [rsp-40h] [rbp-120h]
  unsigned __int64 v34; // [rsp+0h] [rbp-E0h]
  int v35; // [rsp+8h] [rbp-D8h]
  const __m128i *v36; // [rsp+8h] [rbp-D8h]
  __int64 v37; // [rsp+10h] [rbp-D0h]
  unsigned __int8 v38; // [rsp+10h] [rbp-D0h]
  unsigned __int64 v39; // [rsp+10h] [rbp-D0h]
  unsigned __int8 v40; // [rsp+18h] [rbp-C8h]
  __int64 v41; // [rsp+18h] [rbp-C8h]
  __int64 v42; // [rsp+20h] [rbp-C0h]
  __int64 v43; // [rsp+50h] [rbp-90h] BYREF
  int v44; // [rsp+58h] [rbp-88h]
  _BYTE *v45; // [rsp+60h] [rbp-80h] BYREF
  __int64 v46; // [rsp+68h] [rbp-78h]
  _BYTE v47[112]; // [rsp+70h] [rbp-70h] BYREF

  v8 = *(_QWORD **)(a2 + 32);
  v9 = *(_QWORD *)(a2 + 72);
  v10 = v8[15];
  v11 = v8[16];
  v12 = v8[10];
  v13 = *(_QWORD *)(v10 + 40) + 16LL * *((unsigned int *)v8 + 32);
  v14 = v8[11];
  v15 = *(_BYTE *)v13;
  v16 = *(_QWORD *)(v13 + 8);
  v43 = v9;
  if ( v9 )
  {
    v34 = v10;
    v35 = a3;
    v37 = v16;
    v40 = v15;
    sub_1623A60((__int64)&v43, v9, 2);
    v10 = v34;
    a3 = v35;
    v16 = v37;
    v15 = v40;
  }
  v44 = *(_DWORD *)(a2 + 64);
  if ( a3 == 2 )
  {
    v17 = *(_QWORD *)a1;
    if ( v15 && *(_QWORD *)(v17 + 8LL * v15 + 120) )
    {
      v20 = sub_200E230((_QWORD *)a1, v12, v14, v15, v16, a4, a5, *(double *)a6.m128i_i64);
      v21 = *(const __m128i **)(a2 + 32);
      v46 = 0x400000000LL;
      v45 = v47;
      v23 = v22;
      v24 = 40LL * *(unsigned int *)(a2 + 56);
      v25 = (const __m128i *)((char *)v21 + v24);
      v26 = 0xCCCCCCCCCCCCCCCDLL * (v24 >> 3);
      if ( (unsigned __int64)v24 > 0xA0 )
      {
        v36 = (const __m128i *)((char *)v21 + v24);
        v39 = 0xCCCCCCCCCCCCCCCDLL * (v24 >> 3);
        sub_16CD150((__int64)&v45, v47, v26, 16, (int)v47, (int)v25);
        v28 = v46;
        v29 = v45;
        LODWORD(v26) = v39;
        v25 = v36;
        v27 = (__m128i *)&v45[16 * (unsigned int)v46];
      }
      else
      {
        v27 = (__m128i *)v47;
        v28 = 0;
        v29 = v47;
      }
      if ( v21 != v25 )
      {
        do
        {
          if ( v27 )
            *v27 = _mm_loadu_si128(v21);
          v21 = (const __m128i *)((char *)v21 + 40);
          ++v27;
        }
        while ( v25 != v21 );
        v29 = v45;
        v28 = v46;
      }
      v30 = v26 + v28;
      LODWORD(v46) = v30;
      *((_QWORD *)v29 + 4) = v20;
      *((_DWORD *)v29 + 10) = v23;
      v19 = (__int64)sub_1D2E160(*(_QWORD **)(a1 + 8), (__int64 *)a2, (__int64)v29, v30);
      if ( v45 != v47 )
        _libc_free((unsigned __int64)v45);
    }
    else
    {
      v38 = v15;
      v41 = v16;
      sub_1F40D10((__int64)&v45, v17, *(_QWORD *)(*(_QWORD *)(a1 + 8) + 48LL), v15, v16);
      if ( (_BYTE)v45 == 1 )
      {
        v19 = sub_213D210(a1, a2, 3, v18, v41);
      }
      else
      {
        sub_1F40D10((__int64)&v45, *(_QWORD *)a1, *(_QWORD *)(*(_QWORD *)(a1 + 8) + 48LL), v38, v41);
        if ( (_BYTE)v45 == 7 )
          v19 = sub_203DB10((__int64 *)a1, a2, 3, a4, a5, a6);
        else
          v19 = (__int64)sub_202BDB0((__int64 *)a1, a2);
      }
    }
  }
  else
  {
    v42 = sub_2138AD0(a1, v10, v11);
    *((_QWORD *)&v33 + 1) = v14;
    *(_QWORD *)&v33 = v12;
    v19 = sub_1D2C870(
            *(_QWORD **)(a1 + 8),
            **(_QWORD **)(a2 + 32),
            *(_QWORD *)(*(_QWORD *)(a2 + 32) + 8LL),
            (__int64)&v43,
            v42,
            v11 & 0xFFFFFFFF00000000LL | v31,
            *(_OWORD *)(*(_QWORD *)(a2 + 32) + 40LL),
            v33,
            *(unsigned __int8 *)(a2 + 88),
            *(_QWORD *)(a2 + 96),
            *(_QWORD *)(a2 + 104),
            1,
            (*(_BYTE *)(a2 + 27) & 8) != 0);
  }
  if ( v43 )
    sub_161E7C0((__int64)&v43, v43);
  return v19;
}
