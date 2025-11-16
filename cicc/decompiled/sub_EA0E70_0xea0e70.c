// Function: sub_EA0E70
// Address: 0xea0e70
//
const char *__fastcall sub_EA0E70(
        __int64 a1,
        _QWORD *a2,
        _BYTE *a3,
        __int64 a4,
        _BYTE *a5,
        __int64 a6,
        __int128 a7,
        __int128 a8,
        __int128 a9,
        __int128 a10,
        __int64 a11,
        __int64 a12,
        __int64 a13,
        __int64 a14,
        __int64 a15,
        __int64 a16)
{
  _QWORD *v16; // r10
  _BYTE *v17; // r11
  _QWORD *v20; // rdi
  __int64 v21; // r15
  __int64 v22; // r14
  __int64 v23; // rbx
  _BYTE *v24; // r8
  size_t v25; // r9
  __m128i v26; // xmm5
  __m128i v27; // xmm6
  __m128i v28; // xmm7
  __m128i v29; // xmm5
  _DWORD *v30; // rcx
  _DWORD *v31; // rsi
  size_t v32; // rdx
  size_t v33; // r8
  __int64 v34; // r9
  __int64 v36; // rax
  _BYTE *v38; // [rsp+18h] [rbp-B8h]
  _BYTE *v39; // [rsp+18h] [rbp-B8h]
  size_t v40; // [rsp+20h] [rbp-B0h]
  _QWORD *v41; // [rsp+20h] [rbp-B0h]
  __int64 v44; // [rsp+38h] [rbp-98h]
  __int64 v45; // [rsp+40h] [rbp-90h]
  __int64 v46; // [rsp+48h] [rbp-88h]
  __m128i v47; // [rsp+50h] [rbp-80h] BYREF
  __m128i v48; // [rsp+60h] [rbp-70h] BYREF
  __m128i v49; // [rsp+70h] [rbp-60h] BYREF
  __m128i v50; // [rsp+80h] [rbp-50h] BYREF
  size_t v51[7]; // [rsp+98h] [rbp-38h] BYREF

  v16 = a2;
  v17 = a3;
  v20 = (_QWORD *)(a1 + 24);
  v21 = a11;
  v46 = a13;
  v22 = a12;
  v50 = _mm_loadu_si128((const __m128i *)&a7);
  v23 = a14;
  v49 = _mm_loadu_si128((const __m128i *)&a8);
  v45 = a15;
  v48 = _mm_loadu_si128((const __m128i *)&a9);
  v44 = a16;
  v47 = _mm_loadu_si128((const __m128i *)&a10);
  *(v20 - 3) = &unk_49E41D0;
  *(_QWORD *)(a1 + 8) = v20;
  v24 = (_BYTE *)*a2;
  v25 = a2[1];
  if ( v25 + *a2 && !v24 )
    sub_426248((__int64)"basic_string::_M_construct null not valid");
  v51[0] = a2[1];
  if ( v25 > 0xF )
  {
    v38 = v24;
    v40 = v25;
    v36 = sub_22409D0(a1 + 8, v51, 0);
    v25 = v40;
    v24 = v38;
    *(_QWORD *)(a1 + 8) = v36;
    v20 = (_QWORD *)v36;
    v16 = a2;
    v17 = a3;
    *(_QWORD *)(a1 + 24) = v51[0];
    goto LABEL_10;
  }
  if ( v25 != 1 )
  {
    if ( !v25 )
      goto LABEL_6;
LABEL_10:
    v39 = v17;
    v41 = v16;
    memcpy(v20, v24, v25);
    v25 = v51[0];
    v20 = *(_QWORD **)(a1 + 8);
    v17 = v39;
    v16 = v41;
    goto LABEL_6;
  }
  *(_BYTE *)(a1 + 24) = *v24;
LABEL_6:
  *(_QWORD *)(a1 + 16) = v25;
  *((_BYTE *)v20 + v25) = 0;
  *(_QWORD *)(a1 + 40) = v16[4];
  *(_QWORD *)(a1 + 48) = v16[5];
  *(_QWORD *)(a1 + 56) = v16[6];
  *(_QWORD *)(a1 + 64) = a1 + 80;
  sub_E9F6D0((__int64 *)(a1 + 64), v17, (__int64)&v17[a4]);
  *(_QWORD *)(a1 + 96) = a1 + 112;
  sub_E9F6D0((__int64 *)(a1 + 96), a5, (__int64)&a5[a6]);
  v26 = _mm_load_si128(&v49);
  v27 = _mm_load_si128(&v48);
  v28 = _mm_load_si128(&v47);
  *(_QWORD *)(a1 + 184) = v22;
  *(_QWORD *)(a1 + 192) = v46;
  *(__m128i *)(a1 + 128) = v26;
  v29 = _mm_load_si128(&v50);
  v30 = *(_DWORD **)(a1 + 96);
  *(_QWORD *)(a1 + 216) = v45;
  *(_QWORD *)(a1 + 208) = v23;
  v31 = *(_DWORD **)(a1 + 64);
  *(_QWORD *)(a1 + 224) = v44;
  v32 = *(_QWORD *)(a1 + 72);
  *(_QWORD *)(a1 + 176) = v21;
  v33 = *(_QWORD *)(a1 + 104);
  *(_QWORD *)(a1 + 264) = 0;
  *(_QWORD *)(a1 + 272) = a1 + 288;
  *(_QWORD *)(a1 + 280) = 0;
  *(_BYTE *)(a1 + 288) = 0;
  *(__m128i *)(a1 + 144) = v27;
  *(__m128i *)(a1 + 160) = v28;
  *(_OWORD *)(a1 + 232) = 0;
  *(_OWORD *)(a1 + 248) = 0;
  return sub_EA0C90(a1, v31, v32, v30, v33, v34, v29.m128i_i64[0], v29.m128i_i64[1]);
}
