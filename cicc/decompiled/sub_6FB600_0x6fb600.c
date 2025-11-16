// Function: sub_6FB600
// Address: 0x6fb600
//
__int64 __fastcall sub_6FB600(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  char v7; // al
  __int64 v8; // rax
  __int64 v9; // rdi
  __int64 v10; // r13
  bool v11; // zf
  __int64 v12; // rax
  __int64 v13; // r14
  __int64 v14; // rax
  __int64 v15; // rax
  __int64 v16; // r13
  __int64 v17; // rax
  __int64 v18; // rax
  __m128i v20; // [rsp+0h] [rbp-180h] BYREF
  __m128i v21; // [rsp+10h] [rbp-170h]
  __m128i v22; // [rsp+20h] [rbp-160h]
  __m128i v23; // [rsp+30h] [rbp-150h]
  __m128i v24; // [rsp+40h] [rbp-140h]
  __m128i v25; // [rsp+50h] [rbp-130h]
  __m128i v26; // [rsp+60h] [rbp-120h]
  __m128i v27; // [rsp+70h] [rbp-110h]
  __m128i v28; // [rsp+80h] [rbp-100h]
  __m128i v29; // [rsp+90h] [rbp-F0h]
  __m128i v30; // [rsp+A0h] [rbp-E0h]
  __m128i v31; // [rsp+B0h] [rbp-D0h]
  __m128i v32; // [rsp+C0h] [rbp-C0h]
  __m128i v33; // [rsp+D0h] [rbp-B0h]
  __m128i v34; // [rsp+E0h] [rbp-A0h]
  __m128i v35; // [rsp+F0h] [rbp-90h]
  __m128i v36; // [rsp+100h] [rbp-80h]
  __m128i v37; // [rsp+110h] [rbp-70h]
  __m128i v38; // [rsp+120h] [rbp-60h]
  __m128i v39; // [rsp+130h] [rbp-50h]
  __m128i v40; // [rsp+140h] [rbp-40h]
  __m128i v41; // [rsp+150h] [rbp-30h]

  v7 = *(_BYTE *)(a1 + 16);
  v20 = _mm_loadu_si128((const __m128i *)a1);
  v21 = _mm_loadu_si128((const __m128i *)(a1 + 16));
  v22 = _mm_loadu_si128((const __m128i *)(a1 + 32));
  v23 = _mm_loadu_si128((const __m128i *)(a1 + 48));
  v24 = _mm_loadu_si128((const __m128i *)(a1 + 64));
  v25 = _mm_loadu_si128((const __m128i *)(a1 + 80));
  v26 = _mm_loadu_si128((const __m128i *)(a1 + 96));
  v27 = _mm_loadu_si128((const __m128i *)(a1 + 112));
  v28 = _mm_loadu_si128((const __m128i *)(a1 + 128));
  if ( v7 == 2 )
  {
    v29 = _mm_loadu_si128((const __m128i *)(a1 + 144));
    v30 = _mm_loadu_si128((const __m128i *)(a1 + 160));
    v31 = _mm_loadu_si128((const __m128i *)(a1 + 176));
    v32 = _mm_loadu_si128((const __m128i *)(a1 + 192));
    v33 = _mm_loadu_si128((const __m128i *)(a1 + 208));
    v34 = _mm_loadu_si128((const __m128i *)(a1 + 224));
    v35 = _mm_loadu_si128((const __m128i *)(a1 + 240));
    v36 = _mm_loadu_si128((const __m128i *)(a1 + 256));
    v37 = _mm_loadu_si128((const __m128i *)(a1 + 272));
    v38 = _mm_loadu_si128((const __m128i *)(a1 + 288));
    v39 = _mm_loadu_si128((const __m128i *)(a1 + 304));
    v40 = _mm_loadu_si128((const __m128i *)(a1 + 320));
    v41 = _mm_loadu_si128((const __m128i *)(a1 + 336));
  }
  else if ( v7 == 5 || v7 == 1 )
  {
    v29.m128i_i64[0] = *(_QWORD *)(a1 + 144);
  }
  v8 = sub_6F6F40((const __m128i *)a1, 0, a3, a4, a5, a6);
  v9 = *(_QWORD *)v8;
  v10 = v8;
  v11 = *(_BYTE *)(*(_QWORD *)v8 + 140LL) == 12;
  v12 = *(_QWORD *)v8;
  if ( v11 )
  {
    do
      v12 = *(_QWORD *)(v12 + 160);
    while ( *(_BYTE *)(v12 + 140) == 12 );
  }
  v13 = *(_QWORD *)(v12 + 160);
  v14 = sub_72D2E0(v9, 0);
  v15 = sub_73DBF0(0, v14, v10);
  *(_BYTE *)(v15 + 27) |= 2u;
  v16 = v15;
  v17 = sub_72D2E0(v13, 0);
  v18 = sub_73DBF0(5, v17, v16);
  *(_BYTE *)(v18 + 27) |= 2u;
  sub_6E70E0((__int64 *)v18, a1);
  sub_6E4EE0(a1, (__int64)&v20);
  sub_6E5820(*(unsigned __int64 **)(a1 + 88), 32);
  *(_WORD *)(a1 + 18) = *(_WORD *)(a1 + 18) & 0xEFF7 | (((v21.m128i_i8[3] & 0x10) != 0) << 12);
  return sub_6E5070(a1, (__int64)&v20);
}
