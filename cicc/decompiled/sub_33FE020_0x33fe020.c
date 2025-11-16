// Function: sub_33FE020
// Address: 0x33fe020
//
__int64 __fastcall sub_33FE020(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5, char a6, __m128i a7)
{
  unsigned __int64 v8; // r13
  unsigned __int16 v10; // bx
  __int64 v11; // rdx
  __int64 v12; // rax
  __m128i *v13; // rax
  int v14; // ebx
  unsigned int v15; // edx
  __int64 v16; // r9
  __int64 v17; // r9
  __int64 v18; // rax
  int v19; // r8d
  unsigned __int64 v20; // rdx
  __int64 v21; // r8
  __int64 v22; // rax
  __int64 v23; // r14
  __int16 v24; // ax
  __int64 v26; // rdx
  __int64 v27; // rcx
  __int64 v28; // r8
  __int64 v29; // rdi
  int v30; // r9d
  __int64 v31; // rsi
  unsigned __int8 *v32; // rsi
  __int64 v33; // rcx
  unsigned __int64 v34; // rax
  __int64 v35; // rcx
  unsigned __int64 v36; // rax
  __int64 v37; // rax
  int v38; // [rsp+0h] [rbp-120h]
  int v39; // [rsp+0h] [rbp-120h]
  unsigned int v40; // [rsp+8h] [rbp-118h]
  __int64 v41; // [rsp+10h] [rbp-110h]
  __m128i *v42; // [rsp+10h] [rbp-110h]
  __m128i v44; // [rsp+20h] [rbp-100h] BYREF
  __int64 *v45; // [rsp+38h] [rbp-E8h] BYREF
  __m128i v46; // [rsp+40h] [rbp-E0h] BYREF
  unsigned __int8 *v47; // [rsp+50h] [rbp-D0h] BYREF
  int v48; // [rsp+58h] [rbp-C8h]
  _BYTE *v49; // [rsp+60h] [rbp-C0h] BYREF
  __int64 v50; // [rsp+68h] [rbp-B8h]
  _BYTE v51[176]; // [rsp+70h] [rbp-B0h] BYREF

  v8 = a2;
  v10 = a4;
  v44.m128i_i64[0] = a4;
  v44.m128i_i64[1] = a5;
  if ( (_WORD)a4 )
  {
    if ( (unsigned __int16)(a4 - 17) > 0xD3u )
    {
LABEL_3:
      v11 = v44.m128i_i64[1];
      goto LABEL_4;
    }
    v11 = 0;
    v10 = word_4456580[(unsigned __int16)a4 - 1];
  }
  else
  {
    if ( !sub_30070B0((__int64)&v44) )
      goto LABEL_3;
    v10 = sub_3009970((__int64)&v44, a2, v26, v27, v28);
  }
LABEL_4:
  if ( (unsigned int)*(unsigned __int8 *)(*(_QWORD *)(a2 + 8) + 8LL) - 17 <= 1 )
  {
    v41 = v11;
    v12 = sub_AC8EA0(*(__int64 **)(a1 + 64), (__int64 *)(a2 + 24));
    v11 = v41;
    v8 = v12;
  }
  v13 = sub_33ED250(a1, v10, v11);
  v14 = a6 == 0 ? 12 : 36;
  v40 = v15;
  v50 = 0x2000000000LL;
  v42 = v13;
  v49 = v51;
  sub_33C9670((__int64)&v49, v14, (unsigned __int64)v13, 0, 0, v16);
  v18 = (unsigned int)v50;
  v19 = v8;
  v20 = (unsigned int)v50 + 1LL;
  if ( v20 > HIDWORD(v50) )
  {
    sub_C8D5F0((__int64)&v49, v51, v20, 4u, (unsigned int)v8, v17);
    v18 = (unsigned int)v50;
    v19 = v8;
  }
  *(_DWORD *)&v49[4 * v18] = v19;
  v21 = HIDWORD(v8);
  LODWORD(v50) = v50 + 1;
  v22 = (unsigned int)v50;
  if ( (unsigned __int64)(unsigned int)v50 + 1 > HIDWORD(v50) )
  {
    sub_C8D5F0((__int64)&v49, v51, (unsigned int)v50 + 1LL, 4u, v21, v17);
    v22 = (unsigned int)v50;
    v21 = HIDWORD(v8);
  }
  *(_DWORD *)&v49[4 * v22] = v21;
  LODWORD(v50) = v50 + 1;
  v45 = 0;
  v23 = (__int64)sub_33CCCF0(a1, (__int64)&v49, a3, (__int64 *)&v45);
  if ( v23 )
  {
    v24 = v44.m128i_i16[0];
    if ( v44.m128i_i16[0] )
    {
      if ( (unsigned __int16)(v44.m128i_i16[0] - 17) > 0xD3u )
        goto LABEL_13;
      goto LABEL_23;
    }
    if ( !sub_30070B0((__int64)&v44) )
      goto LABEL_13;
LABEL_17:
    v46 = _mm_load_si128(&v44);
    if ( sub_3007100((__int64)&v46) )
      goto LABEL_24;
LABEL_18:
    v23 = sub_32886A0(a1, v46.m128i_u32[0], v46.m128i_i64[1], a3, v23, 0);
    goto LABEL_13;
  }
  v23 = *(_QWORD *)(a1 + 416);
  v29 = a1 + 424;
  if ( (unsigned int)(*(_DWORD *)(*(_QWORD *)(*(_QWORD *)(a1 + 40) + 8LL) + 544LL) - 42) > 1 )
  {
    if ( v23 )
    {
      *(_QWORD *)(a1 + 416) = *(_QWORD *)v23;
    }
    else
    {
      v35 = *(_QWORD *)(a1 + 424);
      *(_QWORD *)(a1 + 504) += 120LL;
      v36 = (v35 + 7) & 0xFFFFFFFFFFFFFFF8LL;
      if ( *(_QWORD *)(a1 + 432) >= v36 + 120 && v35 )
      {
        *(_QWORD *)(a1 + 424) = v36 + 120;
        if ( !v36 )
          goto LABEL_37;
      }
      else
      {
        v36 = sub_9D1E70(v29, 120, 120, 3);
      }
      v23 = v36;
    }
    v47 = 0;
    *(_QWORD *)v23 = 0;
    *(_QWORD *)(v23 + 48) = v42;
    *(_QWORD *)(v23 + 8) = 0;
    *(_QWORD *)(v23 + 16) = 0;
    *(_DWORD *)(v23 + 24) = v14;
    *(_DWORD *)(v23 + 28) = 0;
    *(_WORD *)(v23 + 34) = -1;
    *(_DWORD *)(v23 + 36) = -1;
    *(_QWORD *)(v23 + 40) = 0;
    *(_QWORD *)(v23 + 56) = 0;
    *(_DWORD *)(v23 + 64) = 0;
    *(_QWORD *)(v23 + 68) = v40;
    v32 = v47;
    *(_QWORD *)(v23 + 80) = v47;
    if ( v32 )
    {
LABEL_35:
      sub_B976B0((__int64)&v47, v32, v23 + 80);
      *(_QWORD *)(v23 + 88) = 0xFFFFFFFFLL;
      *(_WORD *)(v23 + 32) = 0;
LABEL_36:
      *(_QWORD *)(v23 + 96) = v8;
      goto LABEL_37;
    }
LABEL_43:
    *(_QWORD *)(v23 + 88) = 0xFFFFFFFFLL;
    *(_WORD *)(v23 + 32) = 0;
    goto LABEL_36;
  }
  v30 = 0;
  if ( !*(_DWORD *)(a1 + 72) )
    v30 = *(_DWORD *)(a3 + 8);
  if ( v23 )
  {
    *(_QWORD *)(a1 + 416) = *(_QWORD *)v23;
    goto LABEL_32;
  }
  v33 = *(_QWORD *)(a1 + 424);
  *(_QWORD *)(a1 + 504) += 120LL;
  v34 = (v33 + 7) & 0xFFFFFFFFFFFFFFF8LL;
  if ( *(_QWORD *)(a1 + 432) < v34 + 120 || !v33 )
  {
    v39 = v30;
    v37 = sub_9D1E70(v29, 120, 120, 3);
    v30 = v39;
    v23 = v37;
LABEL_32:
    v31 = *(_QWORD *)a3;
    v47 = (unsigned __int8 *)v31;
    if ( v31 )
    {
      v38 = v30;
      sub_B96E90((__int64)&v47, v31, 1);
      v30 = v38;
    }
    *(_QWORD *)v23 = 0;
    *(_QWORD *)(v23 + 8) = 0;
    *(_QWORD *)(v23 + 48) = v42;
    *(_QWORD *)(v23 + 16) = 0;
    *(_DWORD *)(v23 + 24) = v14;
    *(_DWORD *)(v23 + 28) = 0;
    *(_WORD *)(v23 + 34) = -1;
    *(_DWORD *)(v23 + 36) = -1;
    *(_QWORD *)(v23 + 40) = 0;
    *(_QWORD *)(v23 + 56) = 0;
    *(_DWORD *)(v23 + 64) = 0;
    *(_DWORD *)(v23 + 68) = v40;
    *(_DWORD *)(v23 + 72) = v30;
    v32 = v47;
    *(_QWORD *)(v23 + 80) = v47;
    if ( v32 )
      goto LABEL_35;
    goto LABEL_43;
  }
  *(_QWORD *)(a1 + 424) = v34 + 120;
  if ( v34 )
  {
    v23 = (v33 + 7) & 0xFFFFFFFFFFFFFFF8LL;
    goto LABEL_32;
  }
LABEL_37:
  sub_C657C0((__int64 *)(a1 + 520), (__int64 *)v23, v45, (__int64)off_4A367D0);
  sub_33CC420(a1, v23);
  v24 = v44.m128i_i16[0];
  if ( !v44.m128i_i16[0] )
  {
    if ( !sub_30070B0((__int64)&v44) )
      goto LABEL_13;
    goto LABEL_17;
  }
  if ( (unsigned __int16)(v44.m128i_i16[0] - 17) > 0xD3u )
    goto LABEL_13;
LABEL_23:
  a7 = _mm_load_si128(&v44);
  v46 = a7;
  if ( (unsigned __int16)(v24 - 176) > 0x34u )
    goto LABEL_18;
LABEL_24:
  if ( *(_DWORD *)(v23 + 24) == 51 )
  {
    v47 = 0;
    v48 = 0;
    v23 = (__int64)sub_33F17F0((_QWORD *)a1, 51, (__int64)&v47, v46.m128i_u32[0], v46.m128i_i64[1]);
    if ( v47 )
      sub_B91220((__int64)&v47, (__int64)v47);
  }
  else
  {
    v23 = (__int64)sub_33FAF80(a1, 168, a3, v46.m128i_i64[0], v46.m128i_i64[1], 0, a7);
  }
LABEL_13:
  if ( v49 != v51 )
    _libc_free((unsigned __int64)v49);
  return v23;
}
