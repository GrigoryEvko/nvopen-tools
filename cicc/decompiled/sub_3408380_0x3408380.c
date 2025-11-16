// Function: sub_3408380
// Address: 0x3408380
//
__m128i *__fastcall sub_3408380(
        __m128i *a1,
        _QWORD *a2,
        __int64 a3,
        __int64 a4,
        __int64 a5,
        __int64 a6,
        __m128i a7,
        __int64 a8)
{
  __int64 v11; // rax
  unsigned __int16 v12; // dx
  __int64 v13; // rcx
  __int64 v14; // rsi
  __int128 v15; // rax
  __int64 v16; // r9
  __int128 v17; // kr00_16
  __m128i v18; // rax
  __int64 v19; // r9
  unsigned __int8 *v20; // rax
  __m128i si128; // xmm0
  __int64 v23; // rdx
  bool v24; // al
  __int64 v25; // rdx
  __int64 v26; // r8
  __m128i v27; // rax
  __int64 v28; // rax
  unsigned __int16 v29; // ax
  __int64 v30; // rdx
  __int64 v31; // r8
  __int128 v32; // [rsp-20h] [rbp-D0h]
  __int128 v33; // [rsp-20h] [rbp-D0h]
  unsigned __int16 v34; // [rsp+4h] [rbp-ACh]
  __int64 v35; // [rsp+8h] [rbp-A8h]
  __int64 v36; // [rsp+8h] [rbp-A8h]
  __int128 v37; // [rsp+20h] [rbp-90h] BYREF
  _QWORD v38[3]; // [rsp+30h] [rbp-80h] BYREF
  __int64 v39; // [rsp+48h] [rbp-68h]
  unsigned int v40; // [rsp+50h] [rbp-60h] BYREF
  __int64 v41; // [rsp+58h] [rbp-58h]
  unsigned __int64 v42; // [rsp+60h] [rbp-50h] BYREF
  __int64 v43; // [rsp+68h] [rbp-48h]

  v11 = *(_QWORD *)(a3 + 48) + 16LL * (unsigned int)a4;
  v38[0] = a5;
  v38[1] = a6;
  v12 = *(_WORD *)v11;
  v13 = *(_QWORD *)(v11 + 8);
  LOWORD(v40) = *(_WORD *)v11;
  v41 = v13;
  if ( (_WORD)a5 )
  {
    v14 = word_4456340[(unsigned __int16)a5 - 1] >> 1;
    if ( (unsigned __int16)(a5 - 17) <= 0x9Eu )
    {
LABEL_3:
      *(_QWORD *)&v15 = sub_3400BD0((__int64)a2, v14, a8, v40, v41, 0, a7, 0);
      v17 = v15;
      goto LABEL_4;
    }
  }
  else
  {
    v34 = v12;
    v35 = v13;
    v39 = sub_3007240((__int64)v38);
    LODWORD(v37) = (unsigned int)v39 >> 1;
    v24 = sub_30070D0((__int64)v38);
    v14 = (unsigned int)v37;
    v13 = v35;
    v12 = v34;
    if ( v24 )
      goto LABEL_3;
  }
  *(_QWORD *)&v37 = (unsigned int)v14;
  if ( v12 )
  {
    if ( (unsigned __int16)(v12 - 17) > 0xD3u )
    {
      LOWORD(v42) = v12;
      v43 = v13;
LABEL_9:
      if ( v12 == 1 || (unsigned __int16)(v12 - 504) <= 7u )
        BUG();
      LODWORD(v43) = *(_QWORD *)&byte_444C4A0[16 * v12 - 16];
      if ( (unsigned int)v43 > 0x40 )
        goto LABEL_12;
LABEL_16:
      v42 = v37;
      goto LABEL_17;
    }
    v12 = word_4456580[v12 - 1];
    v28 = 0;
  }
  else
  {
    v36 = v13;
    if ( !sub_30070B0((__int64)&v40) )
    {
      v43 = v36;
      LOWORD(v42) = 0;
      goto LABEL_15;
    }
    v29 = sub_3009970((__int64)&v40, v14, v25, v36, v26);
    v31 = v30;
    v12 = v29;
    v28 = v31;
  }
  LOWORD(v42) = v12;
  v43 = v28;
  if ( v12 )
    goto LABEL_9;
LABEL_15:
  LODWORD(v43) = sub_3007260((__int64)&v42);
  if ( (unsigned int)v43 <= 0x40 )
    goto LABEL_16;
LABEL_12:
  sub_C43690((__int64)&v42, v37, 0);
LABEL_17:
  v27.m128i_i64[0] = (__int64)sub_3401900((__int64)a2, a8, v40, v41, (__int64)&v42, 1, a7);
  v17 = (__int128)v27;
  if ( (unsigned int)v43 > 0x40 && v42 )
  {
    v37 = (__int128)v27;
    j_j___libc_free_0_0(v42);
    v17 = v37;
  }
LABEL_4:
  *((_QWORD *)&v32 + 1) = a4;
  *(_QWORD *)&v32 = a3;
  v18.m128i_i64[0] = (__int64)sub_3406EB0(a2, 0xB6u, a8, v40, v41, v16, v32, v17);
  *((_QWORD *)&v33 + 1) = a4;
  *(_QWORD *)&v33 = a3;
  v37 = (__int128)v18;
  v20 = sub_3406EB0(a2, 0x55u, a8, v40, v41, v19, v33, v17);
  si128 = _mm_load_si128((const __m128i *)&v37);
  a1[1].m128i_i64[0] = (__int64)v20;
  a1[1].m128i_i64[1] = v23;
  *a1 = si128;
  return a1;
}
