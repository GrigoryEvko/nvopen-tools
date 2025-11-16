// Function: sub_6FB030
// Address: 0x6fb030
//
__int64 __fastcall sub_6FB030(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  __int64 v7; // rdx
  __int64 v8; // rcx
  __int64 v9; // r8
  __int64 v10; // r9
  char v11; // al
  __int64 v12; // rax
  _QWORD *v13; // r14
  __int64 v14; // rdx
  int v15; // ebx
  __int64 v17; // rdi
  __int64 v18; // rdi
  __int64 v19; // rdx
  __int64 v20; // rcx
  __int64 v21; // r8
  __int64 v22; // r9
  __int64 v23; // rax
  __int64 v24; // rax
  __int64 v25; // rdi
  __int64 v26; // [rsp+8h] [rbp-188h] BYREF
  __m128i v27; // [rsp+10h] [rbp-180h] BYREF
  __m128i v28; // [rsp+20h] [rbp-170h]
  __m128i v29; // [rsp+30h] [rbp-160h]
  __m128i v30; // [rsp+40h] [rbp-150h]
  __m128i v31; // [rsp+50h] [rbp-140h]
  __m128i v32; // [rsp+60h] [rbp-130h]
  __m128i v33; // [rsp+70h] [rbp-120h]
  __m128i v34; // [rsp+80h] [rbp-110h]
  __m128i v35; // [rsp+90h] [rbp-100h]
  __m128i v36; // [rsp+A0h] [rbp-F0h]
  __m128i v37; // [rsp+B0h] [rbp-E0h]
  __m128i v38; // [rsp+C0h] [rbp-D0h]
  __m128i v39; // [rsp+D0h] [rbp-C0h]
  __m128i v40; // [rsp+E0h] [rbp-B0h]
  __m128i v41; // [rsp+F0h] [rbp-A0h]
  __m128i v42; // [rsp+100h] [rbp-90h]
  __m128i v43; // [rsp+110h] [rbp-80h]
  __m128i v44; // [rsp+120h] [rbp-70h]
  __m128i v45; // [rsp+130h] [rbp-60h]
  __m128i v46; // [rsp+140h] [rbp-50h]
  __m128i v47; // [rsp+150h] [rbp-40h]
  __m128i v48; // [rsp+160h] [rbp-30h]

  v26 = sub_724DC0(a1, a2, a3, a4, a5, a6);
  v27 = _mm_loadu_si128((const __m128i *)a1);
  v28 = _mm_loadu_si128((const __m128i *)(a1 + 16));
  v11 = *(_BYTE *)(a1 + 16);
  v29 = _mm_loadu_si128((const __m128i *)(a1 + 32));
  v30 = _mm_loadu_si128((const __m128i *)(a1 + 48));
  v31 = _mm_loadu_si128((const __m128i *)(a1 + 64));
  v32 = _mm_loadu_si128((const __m128i *)(a1 + 80));
  v33 = _mm_loadu_si128((const __m128i *)(a1 + 96));
  v34 = _mm_loadu_si128((const __m128i *)(a1 + 112));
  v35 = _mm_loadu_si128((const __m128i *)(a1 + 128));
  if ( v11 == 2 )
  {
    v36 = _mm_loadu_si128((const __m128i *)(a1 + 144));
    v37 = _mm_loadu_si128((const __m128i *)(a1 + 160));
    v38 = _mm_loadu_si128((const __m128i *)(a1 + 176));
    v39 = _mm_loadu_si128((const __m128i *)(a1 + 192));
    v40 = _mm_loadu_si128((const __m128i *)(a1 + 208));
    v41 = _mm_loadu_si128((const __m128i *)(a1 + 224));
    v42 = _mm_loadu_si128((const __m128i *)(a1 + 240));
    v43 = _mm_loadu_si128((const __m128i *)(a1 + 256));
    v44 = _mm_loadu_si128((const __m128i *)(a1 + 272));
    v45 = _mm_loadu_si128((const __m128i *)(a1 + 288));
    v46 = _mm_loadu_si128((const __m128i *)(a1 + 304));
    v47 = _mm_loadu_si128((const __m128i *)(a1 + 320));
    v48 = _mm_loadu_si128((const __m128i *)(a1 + 336));
  }
  else if ( v11 == 5 || v11 == 1 )
  {
    v36.m128i_i64[0] = *(_QWORD *)(a1 + 144);
  }
  v12 = sub_6F6F40((const __m128i *)a1, 0, v7, v8, v9, v10);
  v13 = (_QWORD *)v12;
  if ( *(_BYTE *)(v12 + 24) == 3 )
  {
    v17 = *(_QWORD *)(v12 + 56);
    if ( (*(_BYTE *)(v17 + 176) & 1) != 0 && (*(_BYTE *)(v17 + 170) & 0x10) != 0 && !*(_BYTE *)(v17 + 177) )
      sub_5EB3F0((_QWORD *)v17);
  }
  v14 = qword_4D03C50;
  if ( HIDWORD(qword_4F077B4)
    && (v13[3] & 0x3FF) == 2
    && (*(_BYTE *)(qword_4D03C50 + 16LL) > 3u || (*(_BYTE *)(qword_4D03C50 + 19LL) & 0x40) != 0) )
  {
    v18 = v13[7];
    if ( (*(_BYTE *)(v18 + 171) & 1) != 0 )
    {
      sub_6F8FA0(v18, (_QWORD *)a1);
      sub_6E4BC0(a1, (__int64)&v27);
      v23 = sub_6F6F40((const __m128i *)a1, 0, v19, v20, v21, v22);
      v14 = qword_4D03C50;
      v13 = (_QWORD *)v23;
    }
  }
  if ( (*(_DWORD *)(v14 + 16) & 0x400000FF) == 0x40000001 )
  {
    sub_6E68E0(0x9Du, a1);
    goto LABEL_16;
  }
  if ( word_4D04898 )
  {
LABEL_12:
    v15 = 0;
    goto LABEL_13;
  }
  if ( (*(_BYTE *)(v14 + 18) & 8) == 0 || (*((_BYTE *)v13 + 25) & 3) == 0 )
  {
LABEL_11:
    if ( *(_BYTE *)(v14 + 16) <= 3u && (*(_QWORD *)(v14 + 16) & 0x1000000100LL) == 0x100 )
    {
      sub_6E68E0(0x1Cu, a1);
      goto LABEL_16;
    }
    goto LABEL_12;
  }
  if ( !(unsigned int)sub_717510(v13, v26, 1) )
  {
    v14 = qword_4D03C50;
    goto LABEL_11;
  }
  v24 = sub_8D67C0(*v13);
  v25 = v26;
  *(_BYTE *)(v26 + 168) |= 8u;
  *(_QWORD *)(v25 + 128) = v24;
  sub_6E6A50(v25, a1);
  if ( !*(_BYTE *)(qword_4D03C50 + 16LL) )
    goto LABEL_16;
  v15 = 1;
LABEL_13:
  if ( (unsigned int)sub_8D3410(*v13) )
  {
    v13 = (_QWORD *)sub_6EE5A0((__int64)v13);
    if ( v15 )
      goto LABEL_15;
LABEL_20:
    sub_6E70E0(v13, a1);
    goto LABEL_16;
  }
  *((_BYTE *)v13 + 25) &= 0xFCu;
  if ( !v15 )
    goto LABEL_20;
LABEL_15:
  *(_QWORD *)(a1 + 288) = v13;
LABEL_16:
  sub_6E4EE0(a1, (__int64)&v27);
  sub_6E5820(*(unsigned __int64 **)(a1 + 88), 32);
  *(_WORD *)(a1 + 18) = *(_WORD *)(a1 + 18) & 0xEFD7 | (((v28.m128i_i8[3] & 0x10) != 0) << 12);
  sub_6E5070(a1, (__int64)&v27);
  return sub_724E30(&v26);
}
