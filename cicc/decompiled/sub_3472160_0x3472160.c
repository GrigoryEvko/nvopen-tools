// Function: sub_3472160
// Address: 0x3472160
//
unsigned __int8 *__fastcall sub_3472160(
        __int64 a1,
        __int64 a2,
        __int64 a3,
        __int64 a4,
        __int64 a5,
        __int64 a6,
        __m128i a7,
        __int128 a8,
        unsigned __int64 a9,
        __int64 a10)
{
  __int64 v12; // r15
  unsigned __int16 v13; // r13
  __int64 v14; // rdx
  __int64 v16; // rax
  __int64 v17; // rdx
  unsigned __int16 v18; // r14
  unsigned __int64 v19; // r15
  __int64 v20; // rdx
  char v21; // cl
  __int64 v22; // rax
  __int64 v23; // rsi
  __int64 v24; // rdx
  __int64 v25; // rax
  unsigned __int64 v26; // rdx
  _BOOL8 v27; // rdx
  __int64 v28; // rdx
  __int64 (__fastcall *v29)(__int64, __int64, __int64, __int64, __int64); // rax
  __int64 v30; // rax
  int v31; // edx
  char v32; // al
  char v33; // cl
  int v34; // eax
  __m128i v35; // rax
  unsigned __int64 v36; // rax
  __int64 v37; // r13
  unsigned __int64 v38; // rax
  __int64 v39; // rdi
  __int64 v40; // rax
  unsigned __int8 *v41; // rax
  __int64 v42; // rdx
  char v43; // si
  __int64 v44; // rax
  __int16 v45; // dx
  __m128i *v46; // rbx
  __int64 v47; // rax
  _QWORD *v48; // r13
  __int64 v49; // rdx
  __int64 v50; // r13
  __int64 v51; // rax
  __int64 *v52; // rdi
  __int16 v53; // cx
  __m128i *v54; // r14
  unsigned int v55; // edx
  unsigned int v56; // ebx
  int v57; // r9d
  unsigned int v58; // [rsp+Ch] [rbp-124h]
  unsigned __int8 (__fastcall *v59)(__int64, _QWORD, __int64, _QWORD, __int64, _QWORD, __int64, _QWORD, _QWORD *); // [rsp+10h] [rbp-120h]
  __int64 v60; // [rsp+10h] [rbp-120h]
  unsigned int v61; // [rsp+18h] [rbp-118h]
  char v62; // [rsp+20h] [rbp-110h]
  char v63; // [rsp+20h] [rbp-110h]
  unsigned int v64; // [rsp+20h] [rbp-110h]
  __int64 v65; // [rsp+20h] [rbp-110h]
  __int64 v66; // [rsp+28h] [rbp-108h]
  __int64 v67; // [rsp+40h] [rbp-F0h] BYREF
  __int64 v68; // [rsp+48h] [rbp-E8h]
  __int64 v69; // [rsp+50h] [rbp-E0h] BYREF
  __int64 v70; // [rsp+58h] [rbp-D8h]
  __int64 v71; // [rsp+60h] [rbp-D0h] BYREF
  __int64 v72; // [rsp+68h] [rbp-C8h]
  _QWORD v73[2]; // [rsp+70h] [rbp-C0h] BYREF
  __int64 v74; // [rsp+80h] [rbp-B0h]
  __int64 v75; // [rsp+88h] [rbp-A8h]
  __int64 v76; // [rsp+90h] [rbp-A0h]
  __int64 v77; // [rsp+98h] [rbp-98h]
  __int64 v78; // [rsp+A0h] [rbp-90h]
  __int64 v79; // [rsp+A8h] [rbp-88h]
  __int64 v80; // [rsp+B0h] [rbp-80h]
  __int64 v81; // [rsp+B8h] [rbp-78h]
  __int128 v82; // [rsp+C0h] [rbp-70h]
  __int64 v83; // [rsp+D0h] [rbp-60h]
  __m128i v84; // [rsp+E0h] [rbp-50h] BYREF
  __m128i v85; // [rsp+F0h] [rbp-40h]

  v69 = a2;
  v70 = a3;
  v67 = a5;
  v68 = a6;
  if ( (_WORD)a5 )
  {
    v12 = 0;
    v13 = word_4456580[(unsigned __int16)a5 - 1];
  }
  else
  {
    v13 = sub_3009970((__int64)&v67, a2, a3, a4, a5);
    v12 = v28;
  }
  LOWORD(v71) = v13;
  v72 = v12;
  if ( v13 )
  {
    if ( v13 == 1 || (unsigned __int16)(v13 - 504) <= 7u )
      goto LABEL_68;
    v16 = *(_QWORD *)&byte_444C4A0[16 * v13 - 16];
    if ( !v16 )
      return 0;
  }
  else
  {
    v74 = sub_3007260((__int64)&v71);
    v75 = v14;
    if ( !v74 )
      return 0;
    v16 = sub_3007260((__int64)&v71);
    v76 = v16;
    v77 = v17;
  }
  if ( (v16 & 7) != 0 )
    return 0;
  v18 = v69;
  if ( (_WORD)v69 == v13 )
  {
    if ( v13 )
    {
      v27 = 0;
      goto LABEL_17;
    }
    if ( v70 == v12 )
      return 0;
    v84.m128i_i64[1] = v12;
    v84.m128i_i16[0] = 0;
    goto LABEL_10;
  }
  v84.m128i_i16[0] = v13;
  v84.m128i_i64[1] = v12;
  if ( !v13 )
  {
LABEL_10:
    v80 = sub_3007260((__int64)&v84);
    v19 = v80;
    v81 = v20;
    v21 = v20;
    goto LABEL_11;
  }
  if ( (unsigned __int16)(v13 - 504) <= 7u )
    goto LABEL_68;
  v19 = *(_QWORD *)&byte_444C4A0[16 * v13 - 16];
  v21 = byte_444C4A0[16 * v13 - 8];
LABEL_11:
  if ( v18 )
  {
    if ( v18 != 1 && (unsigned __int16)(v18 - 504) > 7u )
    {
      v26 = *(_QWORD *)&byte_444C4A0[16 * v18 - 16];
      LOBYTE(v25) = byte_444C4A0[16 * v18 - 8];
      goto LABEL_13;
    }
LABEL_68:
    BUG();
  }
  v62 = v21;
  v22 = sub_3007260((__int64)&v69);
  v21 = v62;
  v23 = v22;
  v25 = v24;
  v78 = v23;
  v26 = v23;
  v79 = v25;
LABEL_13:
  v27 = ((_BYTE)v25 || !v21) && v19 < v26;
  if ( !v13 )
    return 0;
LABEL_17:
  if ( !*(_QWORD *)(a1 + 8LL * v13 + 112) || (*(_BYTE *)(a1 + 500LL * v13 + 6712) & 0xFB) != 0 )
    return 0;
  v29 = *(__int64 (__fastcall **)(__int64, __int64, __int64, __int64, __int64))(*(_QWORD *)a1 + 776LL);
  if ( v29 == sub_2FE41D0 )
  {
    if ( (unsigned __int16)(v13 - 17) <= 0xD3u )
    {
      v30 = *(_QWORD *)(a9 + 56);
      if ( !v30 )
        return 0;
      v31 = 1;
      do
      {
        if ( !*(_DWORD *)(v30 + 8) )
        {
          if ( !v31 )
            return 0;
          v30 = *(_QWORD *)(v30 + 32);
          if ( !v30 )
            goto LABEL_38;
          if ( !*(_DWORD *)(v30 + 8) )
            return 0;
          v31 = 0;
        }
        v30 = *(_QWORD *)(v30 + 32);
      }
      while ( v30 );
      if ( v31 == 1 )
        return 0;
    }
  }
  else if ( !(unsigned __int8)v29(a1, a9, v27, (unsigned int)v71, v72) )
  {
    return 0;
  }
LABEL_38:
  v32 = sub_2EAC4F0(*(_QWORD *)(a9 + 112));
  BYTE4(v83) = 0;
  v33 = v32;
  v82 = 0u;
  v34 = *(_DWORD *)(a8 + 24);
  LODWORD(v83) = 0;
  if ( v34 == 35 || v34 == 11 )
  {
    v47 = *(_QWORD *)(a8 + 96);
    v48 = *(_QWORD **)(v47 + 24);
    if ( *(_DWORD *)(v47 + 32) > 0x40u )
      v48 = (_QWORD *)*v48;
    v63 = v33;
    v73[0] = sub_2D5B750((unsigned __int16 *)&v71);
    v73[1] = v49;
    v84.m128i_i64[0] = v73[0] * (int)v48;
    v84.m128i_i8[8] = v49;
    v50 = (unsigned int)((unsigned __int64)sub_CA1930(&v84) >> 3);
    sub_327C6E0((__int64)&v84, *(__int64 **)(a9 + 112), v50);
    a7 = _mm_loadu_si128(&v84);
    LODWORD(v83) = v85.m128i_i32[0];
    v82 = (__int128)a7;
    BYTE4(v83) = v85.m128i_i8[4];
    v36 = v50;
  }
  else
  {
    v63 = v33;
    LODWORD(v83) = sub_2EAC1E0(*(_QWORD *)(a9 + 112));
    v35.m128i_i64[0] = sub_2D5B750((unsigned __int16 *)&v71);
    v84 = v35;
    v36 = (unsigned __int64)sub_CA1930(&v84) >> 3;
  }
  v37 = 0xFFFFFFFFLL;
  v38 = -(__int64)(v36 | (1LL << v63)) & (v36 | (1LL << v63));
  if ( v38 )
  {
    _BitScanReverse64(&v38, v38);
    v37 = 63 - ((unsigned int)v38 ^ 0x3F);
  }
  LODWORD(v73[0]) = 0;
  v39 = *(_QWORD *)(a9 + 112);
  v59 = *(unsigned __int8 (__fastcall **)(__int64, _QWORD, __int64, _QWORD, __int64, _QWORD, __int64, _QWORD, _QWORD *))(*(_QWORD *)a1 + 824LL);
  v61 = *(unsigned __int16 *)(v39 + 32);
  v64 = sub_2EAC1E0(v39);
  v40 = sub_2E79000(*(__int64 **)(a10 + 40));
  if ( !v59(a1, *(_QWORD *)(a10 + 64), v40, (unsigned int)v71, v72, v64, v37, v61, v73) || !LODWORD(v73[0]) )
    return 0;
  v41 = sub_3466750(
          a1,
          (_QWORD *)a10,
          *(_QWORD *)(*(_QWORD *)(a9 + 40) + 40LL),
          *(_QWORD *)(*(_QWORD *)(a9 + 40) + 48LL),
          (unsigned int)v67,
          v68,
          a7,
          a8);
  v66 = v42;
  v58 = v71;
  v60 = v72;
  v65 = (__int64)v41;
  if ( sub_3280A00((__int64)&v69, (unsigned int)v71, v72) )
  {
    if ( (_WORD)v71 && v18 )
      v43 = (int)*(unsigned __int16 *)(a1 + 2 * ((unsigned __int16)v71 + 274LL * v18 + 71704) + 6) >> 12 == 0 ? 3 : 1;
    else
      v43 = 1;
    v44 = *(_QWORD *)(a9 + 112);
    v84 = _mm_loadu_si128((const __m128i *)(v44 + 40));
    v85 = _mm_loadu_si128((const __m128i *)(v44 + 56));
    v45 = *(_WORD *)(v44 + 32);
    BYTE1(v44) = 1;
    LOBYTE(v44) = v37;
    v46 = sub_33F1DB0(
            (__int64 *)a10,
            v43,
            a4,
            v69,
            v70,
            v44,
            *(_OWORD *)*(_QWORD *)(a9 + 40),
            v65,
            v66,
            v82,
            v83,
            v71,
            v72,
            v45,
            (__int64)&v84);
    sub_3417D40((_QWORD *)a10, a9, (__int64)v46);
  }
  else
  {
    v51 = *(_QWORD *)(a9 + 112);
    v52 = *(__int64 **)(a9 + 40);
    v84 = _mm_loadu_si128((const __m128i *)(v51 + 40));
    v85 = _mm_loadu_si128((const __m128i *)(v51 + 56));
    v53 = *(_WORD *)(v51 + 32);
    BYTE1(v51) = 1;
    LOBYTE(v51) = v37;
    v54 = sub_33F1F00((__int64 *)a10, v58, v60, a4, *v52, v52[1], v65, v66, v82, v83, v51, v53, (__int64)&v84, 0);
    v56 = v55;
    sub_3417D40((_QWORD *)a10, a9, (__int64)v54);
    if ( sub_3280B30((__int64)&v69, (unsigned int)v71, v72) )
      return sub_33FAF80(a10, 216, a4, (unsigned int)v69, v70, v57, a7);
    else
      return sub_33FB890(a10, v69, v70, (__int64)v54, v56, a7);
  }
  return (unsigned __int8 *)v46;
}
