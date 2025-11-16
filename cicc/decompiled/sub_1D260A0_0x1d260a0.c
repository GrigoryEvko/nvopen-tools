// Function: sub_1D260A0
// Address: 0x1d260a0
//
__int64 __fastcall sub_1D260A0(
        _QWORD *a1,
        int a2,
        char a3,
        __int64 a4,
        __int64 a5,
        __int64 a6,
        __int128 a7,
        __int64 a8,
        __int64 a9,
        __int128 a10,
        __int64 a11,
        __int64 a12,
        __int64 a13)
{
  unsigned __int8 v13; // r14
  __int64 v14; // r13
  __int64 v15; // r12
  unsigned __int8 *v16; // rax
  int v17; // edx
  __m128i v18; // xmm0
  __m128i v19; // xmm1
  __int64 **v20; // rbx
  __int64 *v21; // rsi
  __int64 v22; // rsi
  __int16 v23; // bx
  unsigned __int16 v24; // bx
  int v25; // eax
  _QWORD *v26; // rax
  __int64 v27; // rdx
  __int64 v28; // rcx
  __int64 v29; // r8
  __int64 v30; // r9
  __int64 v31; // rbx
  int v33; // edx
  int v34; // r12d
  __int16 v35; // ax
  __int128 v36; // [rsp-20h] [rbp-1D0h]
  __int128 v37; // [rsp-20h] [rbp-1D0h]
  int v38; // [rsp+18h] [rbp-198h]
  __int64 v39; // [rsp+20h] [rbp-190h]
  unsigned __int8 v40; // [rsp+2Ah] [rbp-186h]
  char v41; // [rsp+2Bh] [rbp-185h]
  char v42; // [rsp+2Ch] [rbp-184h]
  __int16 v43; // [rsp+2Ch] [rbp-184h]
  __int64 v44; // [rsp+30h] [rbp-180h]
  unsigned __int8 *v46; // [rsp+48h] [rbp-168h] BYREF
  __m128i v47; // [rsp+50h] [rbp-160h] BYREF
  __int64 v48; // [rsp+60h] [rbp-150h]
  __int64 v49; // [rsp+68h] [rbp-148h]
  __m128i v50; // [rsp+70h] [rbp-140h]
  __int64 *v51[3]; // [rsp+80h] [rbp-130h] BYREF
  __int16 v52; // [rsp+9Ah] [rbp-116h]
  __int64 v53[5]; // [rsp+C8h] [rbp-E8h] BYREF
  unsigned __int64 v54[2]; // [rsp+F0h] [rbp-C0h] BYREF
  _BYTE v55[176]; // [rsp+100h] [rbp-B0h] BYREF

  v13 = a11;
  v14 = a8;
  v42 = a2;
  v40 = a11;
  v15 = a9;
  v39 = a12;
  if ( (_BYTE)a4 == (_BYTE)a11 && ((v41 = 0, (_BYTE)a11) || a12 == a5) )
  {
    if ( !a2 )
    {
LABEL_20:
      v44 = sub_1D252B0((__int64)a1, a4, a5, 1, 0);
      v38 = v33;
      goto LABEL_5;
    }
  }
  else
  {
    v41 = a3 & 3;
    if ( !a2 )
      goto LABEL_20;
  }
  v16 = (unsigned __int8 *)(*(_QWORD *)(a8 + 40) + 16LL * (unsigned int)a9);
  v44 = sub_1D25E70((__int64)a1, a4, a5, *v16, *((_QWORD *)v16 + 1), a6, 1, 0);
  v38 = v17;
LABEL_5:
  v18 = _mm_loadu_si128((const __m128i *)&a7);
  v19 = _mm_loadu_si128((const __m128i *)&a10);
  v49 = v15;
  v54[0] = (unsigned __int64)v55;
  v54[1] = 0x2000000000LL;
  v48 = v14;
  v47 = v18;
  v50 = v19;
  sub_16BD430((__int64)v54, 185);
  sub_16BD4C0((__int64)v54, v44);
  v20 = (__int64 **)&v47;
  do
  {
    v21 = *v20;
    v20 += 2;
    sub_16BD4C0((__int64)v54, (__int64)v21);
    sub_16BD430((__int64)v54, *((_DWORD *)v20 - 2));
  }
  while ( v20 != v51 );
  v22 = v13;
  if ( !v13 )
    v22 = v39;
  sub_16BD4D0((__int64)v54, v22);
  v46 = 0;
  *((_QWORD *)&v36 + 1) = v39;
  *(_QWORD *)&v36 = v40;
  sub_1D189E0((__int64)v51, 185, *(_DWORD *)(a6 + 8), &v46, v44, v38, v36, a13);
  v43 = v42 & 7;
  v23 = v52 & 0xFC7F | (v43 << 7);
  LOBYTE(v52) = v52 & 0x7F | ((_BYTE)v43 << 7);
  HIBYTE(v52) = (4 * v41) | HIBYTE(v23) & 0xF3;
  v24 = v52 & 0xFFFA;
  if ( v53[0] )
    sub_161E7C0((__int64)v53, v53[0]);
  if ( v46 )
    sub_161E7C0((__int64)&v46, (__int64)v46);
  sub_16BD3E0((__int64)v54, v24);
  v25 = sub_1E340A0(a13);
  sub_16BD430((__int64)v54, v25);
  v51[0] = 0;
  v26 = sub_1D17920((__int64)a1, (__int64)v54, a6, (__int64 *)v51);
  v31 = (__int64)v26;
  if ( v26 )
  {
    sub_1E34340(v26[13], a13, v27, v28, v29, v30);
  }
  else
  {
    v31 = a1[26];
    v34 = *(_DWORD *)(a6 + 8);
    if ( v31 )
      a1[26] = *(_QWORD *)v31;
    else
      v31 = sub_145CBF0(a1 + 27, 112, 8);
    *((_QWORD *)&v37 + 1) = v39;
    *(_QWORD *)&v37 = v40;
    sub_1D189E0(v31, 185, v34, (unsigned __int8 **)a6, v44, v38, v37, a13);
    v35 = (v43 << 7) | *(_WORD *)(v31 + 26) & 0xFC7F;
    *(_WORD *)(v31 + 26) = v35;
    *(_BYTE *)(v31 + 27) = HIBYTE(v35) & 0xF3 | (4 * v41);
    sub_1D23B60((__int64)a1, v31, (__int64)&v47, 3);
    sub_16BDA20(a1 + 40, (__int64 *)v31, v51[0]);
    sub_1D172A0((__int64)a1, v31);
  }
  if ( (_BYTE *)v54[0] != v55 )
    _libc_free(v54[0]);
  return v31;
}
