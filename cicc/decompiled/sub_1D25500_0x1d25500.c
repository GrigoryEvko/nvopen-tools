// Function: sub_1D25500
// Address: 0x1d25500
//
_QWORD *__fastcall sub_1D25500(
        _QWORD *a1,
        __int64 a2,
        __int64 a3,
        __int64 a4,
        __int64 a5,
        __int64 a6,
        __int128 a7,
        char a8)
{
  unsigned __int8 *v12; // rax
  __int64 v13; // rax
  __m128i v14; // xmm2
  __int64 v15; // rax
  int v16; // edx
  __m128i v17; // xmm0
  __m128i v18; // xmm1
  unsigned __int64 *v19; // r15
  __int64 v20; // rsi
  __int64 v21; // rsi
  int v22; // eax
  _QWORD *v23; // rax
  _QWORD *v24; // r12
  __int64 v26; // rcx
  unsigned __int8 v27; // r8
  __int64 v28; // r9
  __int64 v29; // rax
  __int64 v30; // rbx
  int v31; // r10d
  __int16 v32; // r14
  __int64 v33; // rax
  __int128 v34; // [rsp-20h] [rbp-170h]
  __int64 v35; // [rsp+8h] [rbp-148h]
  unsigned __int8 v36; // [rsp+13h] [rbp-13Dh]
  int v37; // [rsp+14h] [rbp-13Ch]
  __int64 v38; // [rsp+18h] [rbp-138h]
  int v39; // [rsp+28h] [rbp-128h]
  __int64 v40; // [rsp+38h] [rbp-118h]
  __int64 *v41; // [rsp+48h] [rbp-108h] BYREF
  _OWORD v42[2]; // [rsp+50h] [rbp-100h] BYREF
  __int64 v43; // [rsp+70h] [rbp-E0h]
  __int64 v44; // [rsp+78h] [rbp-D8h]
  __m128i v45; // [rsp+80h] [rbp-D0h]
  unsigned __int64 v46[2]; // [rsp+90h] [rbp-C0h] BYREF
  _BYTE v47[176]; // [rsp+A0h] [rbp-B0h] BYREF

  v12 = (unsigned __int8 *)(*(_QWORD *)(a5 + 40) + 16LL * (unsigned int)a6);
  v13 = sub_1D252B0((__int64)a1, *v12, *((_QWORD *)v12 + 1), 1, 0);
  v14 = _mm_loadu_si128((const __m128i *)&a7);
  v40 = v13;
  v15 = *(_QWORD *)(a2 + 32);
  v39 = v16;
  v17 = _mm_loadu_si128((const __m128i *)v15);
  v18 = _mm_loadu_si128((const __m128i *)(v15 + 40));
  v44 = a6;
  v43 = a5;
  v46[0] = (unsigned __int64)v47;
  v46[1] = 0x2000000000LL;
  v42[0] = v17;
  v42[1] = v18;
  v45 = v14;
  sub_16BD430((__int64)v46, 186);
  sub_16BD4C0((__int64)v46, v40);
  v19 = (unsigned __int64 *)v42;
  do
  {
    v20 = *v19;
    v19 += 2;
    sub_16BD4C0((__int64)v46, v20);
    sub_16BD430((__int64)v46, *((_DWORD *)v19 - 2));
  }
  while ( v19 != v46 );
  v21 = *(unsigned __int8 *)(a2 + 88);
  if ( !(_BYTE)v21 )
    v21 = *(_QWORD *)(a2 + 96);
  sub_16BD4D0((__int64)v19, v21);
  sub_16BD430((__int64)v19, *(_WORD *)(a2 + 26) & 0xFFFA);
  v22 = sub_1E340A0(*(_QWORD *)(a2 + 104));
  sub_16BD430((__int64)v19, v22);
  v41 = 0;
  v23 = sub_1D17920((__int64)a1, (__int64)v19, a4, (__int64 *)&v41);
  if ( v23 )
  {
    v24 = v23;
  }
  else
  {
    v26 = *(_QWORD *)(a2 + 104);
    v27 = *(_BYTE *)(a2 + 88);
    v28 = *(_QWORD *)(a2 + 96);
    v29 = *(_BYTE *)(a2 + 27) >> 2;
    v30 = a1[26];
    v31 = *(_DWORD *)(a4 + 8);
    v32 = v29 & 1;
    if ( v30 )
    {
      a1[26] = *(_QWORD *)v30;
    }
    else
    {
      v36 = v27;
      v35 = v26;
      v37 = *(_DWORD *)(a4 + 8);
      v38 = v28;
      v33 = sub_145CBF0(a1 + 27, 112, 8);
      v28 = v38;
      v31 = v37;
      v26 = v35;
      v27 = v36;
      v30 = v33;
    }
    *((_QWORD *)&v34 + 1) = v28;
    *(_QWORD *)&v34 = v27;
    sub_1D189E0(v30, 186, v31, (unsigned __int8 **)a4, v40, v39, v34, v26);
    *(_WORD *)(v30 + 26) = *(_WORD *)(v30 + 26) & 0xF87F | (v32 << 10) | ((a8 & 7) << 7);
    sub_1D23B60((__int64)a1, v30, (__int64)v42, 4);
    sub_16BDA20(a1 + 40, (__int64 *)v30, v41);
    v24 = (_QWORD *)v30;
    sub_1D172A0((__int64)a1, v30);
  }
  if ( (_BYTE *)v46[0] != v47 )
    _libc_free(v46[0]);
  return v24;
}
