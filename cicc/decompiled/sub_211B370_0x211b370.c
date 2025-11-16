// Function: sub_211B370
// Address: 0x211b370
//
__int64 __fastcall sub_211B370(__int64 a1, unsigned __int64 a2, unsigned int a3, double a4, double a5, double a6)
{
  unsigned __int8 *v8; // rdx
  __int64 v9; // r13
  __int64 v10; // r14
  __int64 v11; // rsi
  unsigned __int8 *v12; // rax
  const void **v13; // r14
  __int64 v14; // rsi
  __int64 v15; // r9
  __int64 v16; // r8
  __int64 v17; // rdi
  _QWORD *v18; // r11
  int v19; // r14d
  __int64 v20; // rax
  __int64 v21; // rax
  const __m128i *v22; // r9
  __int64 v23; // rbx
  __int64 v24; // r12
  __m128i v26; // xmm1
  int v27; // r9d
  __int64 v28; // rax
  unsigned int v29; // edx
  const __m128i *v30; // r9
  __int64 v31; // rdx
  __int128 v32; // [rsp-10h] [rbp-B0h]
  __int64 v33; // [rsp+0h] [rbp-A0h]
  _QWORD *v34; // [rsp+0h] [rbp-A0h]
  __int64 v35; // [rsp+8h] [rbp-98h]
  unsigned __int8 v36; // [rsp+17h] [rbp-89h]
  bool v37; // [rsp+18h] [rbp-88h]
  _QWORD *v38; // [rsp+18h] [rbp-88h]
  __int64 v39; // [rsp+20h] [rbp-80h]
  unsigned int v40; // [rsp+20h] [rbp-80h]
  __int64 v41; // [rsp+28h] [rbp-78h]
  __int64 v42; // [rsp+28h] [rbp-78h]
  __int64 v43; // [rsp+30h] [rbp-70h]
  __int64 v44; // [rsp+40h] [rbp-60h] BYREF
  int v45; // [rsp+48h] [rbp-58h]
  __m128i v46; // [rsp+50h] [rbp-50h] BYREF
  __int64 v47; // [rsp+60h] [rbp-40h]

  v8 = (unsigned __int8 *)(*(_QWORD *)(a2 + 40) + 16LL * a3);
  v9 = *v8;
  v10 = *((_QWORD *)v8 + 1);
  sub_1F40D10((__int64)&v46, *(_QWORD *)a1, *(_QWORD *)(*(_QWORD *)(a1 + 8) + 48LL), (unsigned __int8)v9, v10);
  if ( (_BYTE)v9 == v46.m128i_i8[8] )
  {
    v11 = *(_QWORD *)a1;
    v37 = (_BYTE)v9 == 0 && v10 != v47;
    if ( v37 )
    {
      v37 = 0;
    }
    else if ( (_BYTE)v9 )
    {
      v37 = *(_QWORD *)(v11 + 8 * v9 + 120) != 0;
    }
  }
  else
  {
    v37 = 0;
    v11 = *(_QWORD *)a1;
  }
  v12 = *(unsigned __int8 **)(a2 + 40);
  v13 = (const void **)*((_QWORD *)v12 + 1);
  v36 = *v12;
  sub_1F40D10((__int64)&v46, v11, *(_QWORD *)(*(_QWORD *)(a1 + 8) + 48LL), *v12, (__int64)v13);
  v14 = *(_QWORD *)(a2 + 72);
  v15 = v47;
  v16 = v46.m128i_u8[8];
  v44 = v14;
  if ( v14 )
  {
    v39 = v46.m128i_u8[8];
    v41 = v47;
    sub_1623A60((__int64)&v44, v14, 2);
    v16 = v39;
    v15 = v41;
  }
  v17 = *(_QWORD *)(a2 + 104);
  v33 = v16;
  v35 = v15;
  v18 = *(_QWORD **)(a1 + 8);
  v45 = *(_DWORD *)(a2 + 64);
  v40 = *(_WORD *)(v17 + 32) & 0x1CF;
  if ( (*(_BYTE *)(a2 + 27) & 0xC) != 0 )
  {
    v26 = _mm_loadu_si128((const __m128i *)(v17 + 40));
    v34 = v18;
    v46 = v26;
    v47 = *(_QWORD *)(v17 + 56);
    v27 = sub_1E34390(v17);
    v28 = *(_QWORD *)(a2 + 32);
    v43 = sub_1D264C0(
            v34,
            (*(_WORD *)(a2 + 26) >> 7) & 7,
            0,
            *(unsigned __int8 *)(a2 + 88),
            *(_QWORD *)(a2 + 96),
            (__int64)&v44,
            *(_OWORD *)v28,
            *(_QWORD *)(v28 + 40),
            *(_QWORD *)(v28 + 48),
            *(_OWORD *)(v28 + 80),
            *(_OWORD *)*(_QWORD *)(a2 + 104),
            *(_QWORD *)(*(_QWORD *)(a2 + 104) + 16LL),
            *(unsigned __int8 *)(a2 + 88),
            *(_QWORD *)(a2 + 96),
            v27,
            v40,
            (__int64)&v46,
            0);
    v42 = v29;
    sub_2013400(a1, a2, 1, v43, (__m128i *)1, v30);
    *((_QWORD *)&v32 + 1) = v42;
    *(_QWORD *)&v32 = v43;
    v24 = sub_1D309E0(*(__int64 **)(a1 + 8), 157, (__int64)&v44, v36, v13, 0, a4, *(double *)v26.m128i_i64, a6, v32);
    if ( !v37 )
      v24 = sub_200D2A0(a1, v24, v31, a4, *(double *)v26.m128i_i64, a6);
  }
  else
  {
    v38 = v18;
    v46 = _mm_loadu_si128((const __m128i *)(v17 + 40));
    v47 = *(_QWORD *)(v17 + 56);
    v19 = sub_1E34390(v17);
    v20 = *(_QWORD *)(a2 + 32);
    v21 = sub_1D264C0(
            v38,
            (*(_WORD *)(a2 + 26) >> 7) & 7,
            (*(_BYTE *)(a2 + 27) >> 2) & 3,
            v33,
            v35,
            (__int64)&v44,
            *(_OWORD *)v20,
            *(_QWORD *)(v20 + 40),
            *(_QWORD *)(v20 + 48),
            *(_OWORD *)(v20 + 80),
            *(_OWORD *)*(_QWORD *)(a2 + 104),
            *(_QWORD *)(*(_QWORD *)(a2 + 104) + 16LL),
            v33,
            v35,
            v19,
            v40,
            (__int64)&v46,
            0);
    v23 = v21;
    if ( a2 != v21 )
      sub_2013400(a1, a2, 1, v21, (__m128i *)1, v22);
    v24 = v23;
  }
  if ( v44 )
    sub_161E7C0((__int64)&v44, v44);
  return v24;
}
