// Function: sub_7FDF40
// Address: 0x7fdf40
//
__m128i *__fastcall sub_7FDF40(__int64 a1, char a2, int a3)
{
  unsigned __int8 v4; // al
  __m128i *v5; // r13
  _QWORD *v6; // rdx
  char v7; // al
  __int64 v9; // r14
  __int64 v10; // rax
  __int64 i; // r15
  __int64 v12; // r14
  __m128i *v13; // r13
  __int64 v14; // rcx
  char v15; // al
  unsigned __int8 v16; // r15
  _QWORD *v17; // rax
  __int64 v18; // rsi
  _QWORD *v19; // rdx
  _QWORD *v20; // rax
  char v21; // al
  char v22; // cl
  __int8 v23; // al
  __int32 v24; // eax
  __int8 v25; // dl
  __int8 v26; // al
  __int8 v27; // al
  __int8 v28; // al
  __int8 v29; // dl
  __int8 v30; // dl
  __int8 v31; // al
  __int8 v32; // al
  char v33; // cl
  __int64 v34; // r15
  __int16 v35; // ax
  __int8 v36; // al
  __int8 v37; // al
  _QWORD *v38; // r14
  _QWORD *v39; // rax
  _QWORD *v40; // rax
  __int8 v41; // al
  _QWORD *v42; // rsi
  __int64 v43; // rax
  _QWORD *v44; // rax
  __int64 v45; // [rsp+8h] [rbp-48h]
  _QWORD *v46; // [rsp+8h] [rbp-48h]
  __int64 v47; // [rsp+8h] [rbp-48h]
  char v48; // [rsp+10h] [rbp-40h]
  _QWORD *v49; // [rsp+10h] [rbp-40h]

  v4 = *(_BYTE *)(a1 + 205);
  if ( (v4 & 0x1C) == 0 )
  {
    sub_7FA1F0(a1);
    v4 = *(_BYTE *)(a1 + 205);
  }
  v5 = (__m128i *)a1;
  if ( ((v4 >> 2) & 7) != a2 )
  {
    v6 = *(_QWORD **)(a1 + 176);
    if ( v6 )
    {
      while ( 1 )
      {
        v5 = (__m128i *)v6[1];
        if ( a2 == (((unsigned __int8)v5[12].m128i_i8[13] >> 2) & 7) )
          break;
        v6 = (_QWORD *)*v6;
        if ( !v6 )
          goto LABEL_27;
      }
      v7 = *(_BYTE *)(a1 + 198);
      if ( (v7 & 8) == 0 )
        goto LABEL_10;
LABEL_9:
      v5[12].m128i_i8[6] |= 8u;
      v7 = *(_BYTE *)(a1 + 198);
LABEL_10:
      if ( (v7 & 0x10) != 0 )
      {
        v5[12].m128i_i8[6] |= 0x10u;
        v7 = *(_BYTE *)(a1 + 198);
      }
      if ( (v7 & 0x20) != 0 )
      {
        v5[12].m128i_i8[6] |= 0x20u;
        v7 = *(_BYTE *)(a1 + 198);
      }
      if ( (v7 & 0x40) != 0 )
        v5[12].m128i_i8[6] |= 0x40u;
      if ( (*(_QWORD *)(a1 + 192) & 0x240000000LL) != 0 )
        v5[12].m128i_i8[4] |= 2u;
      if ( *(_DWORD *)(a1 + 160) && a3 )
      {
        v9 = a1;
        if ( a2 == 3 )
          v9 = sub_7FDF40(a1, 1, 0);
        if ( v5[10].m128i_i8[12] == 1 )
          v5[10].m128i_i8[12] = *(_BYTE *)(a1 + 172);
        sub_736C90((__int64)v5, *(_BYTE *)(a1 + 192) >> 7);
        v5[12].m128i_i8[11] = *(_BYTE *)(a1 + 203) & 0x40 | v5[12].m128i_i8[11] & 0xBF;
        sub_7FCF80(v9, (__int64)v5, 0);
        if ( (*(_BYTE *)(a1 + 205) & 2) != 0 )
          sub_7E5120((__int64)v5);
      }
      return v5;
    }
LABEL_27:
    v10 = *(_QWORD *)(a1 + 152);
    for ( i = v10; *(_BYTE *)(i + 140) == 12; i = *(_QWORD *)(i + 160) )
      ;
    v12 = *(_QWORD *)(v10 + 168);
    v13 = *(__m128i **)(*(_QWORD *)(i + 168) + 40LL);
    if ( v13 )
      v13 = (__m128i *)sub_8D71D0(i);
    if ( !(unsigned int)sub_7E4C00(*(_QWORD *)(v12 + 8), i) )
      v13 = sub_73C570(v13, 1);
    v14 = sub_7F8700(i);
    if ( unk_4F06878 && (unsigned __int8)(a2 - 3) <= 1u && *(_BYTE *)(a1 + 174) == 2 )
      v14 = sub_72CBE0();
    v15 = 1;
    v45 = v14;
    v16 = *(_BYTE *)(a1 + 172);
    if ( v16 )
      v15 = *(_BYTE *)(a1 + 172);
    v48 = v15;
    v17 = sub_7259C0(7);
    v18 = v17[21];
    v19 = v17;
    v17[20] = v45;
    *(_BYTE *)(v18 + 16) = (2 * (dword_4F06968 == 0)) | *(_BYTE *)(v18 + 16) & 0xFD;
    if ( v13 )
    {
      v46 = v17;
      v20 = sub_724EF0((__int64)v13);
      v19 = v46;
      *(_QWORD *)v46[21] = v20;
    }
    v47 = (__int64)v19;
    v5 = sub_725FD0();
    v5[10].m128i_i8[12] = v48;
    v21 = 3;
    if ( v16 > 1u )
      v21 = v16 == 2;
    v5[9].m128i_i64[1] = v47;
    v22 = 16 * v21;
    v23 = v5[5].m128i_i8[8];
    v5[12].m128i_i8[1] |= 0x10u;
    v5[5].m128i_i8[8] = v22 | v23 & 0x8F;
    sub_736C90((__int64)v5, *(_BYTE *)(a1 + 192) >> 7);
    v24 = *(_DWORD *)(a1 + 212);
    v5[5].m128i_i8[9] |= 4u;
    v5[13].m128i_i32[1] = v24;
    v5[2].m128i_i64[1] = *(_QWORD *)(a1 + 40);
    sub_725ED0((__int64)v5, *(_BYTE *)(a1 + 174));
    v5[20].m128i_i64[0] = a1;
    v25 = v5[12].m128i_i8[0];
    v5[12].m128i_i8[13] = (4 * (a2 & 7)) | v5[12].m128i_i8[13] & 0xE3;
    v26 = v5[12].m128i_i8[1] | 0x10;
    v5[12].m128i_i8[1] = v26;
    v5[12].m128i_i8[0] = *(_BYTE *)(a1 + 192) & 8 | v25 & 0xF7;
    v5[12].m128i_i8[14] = *(_BYTE *)(a1 + 206) & 0x10 | v5[12].m128i_i8[14] & 0xEF;
    v27 = *(_BYTE *)(a1 + 193) & 1 | v26 & 0xFE;
    v5[12].m128i_i8[1] = v27;
    v28 = *(_BYTE *)(a1 + 193) & 2 | v27 & 0xFD;
    v5[12].m128i_i8[1] = v28;
    v29 = v5[12].m128i_i8[2];
    v5[12].m128i_i8[1] = *(_BYTE *)(a1 + 193) & 4 | v28 & 0xFB;
    v30 = *(_BYTE *)(a1 + 194) & 0x20 | v29 & 0xDF;
    v31 = v5[12].m128i_i8[3];
    v5[12].m128i_i8[2] = v30;
    v32 = *(_BYTE *)(a1 + 195) & 0x40 | v31 & 0xBF;
    v5[12].m128i_i8[3] = v32;
    v5[12].m128i_i8[3] = *(_BYTE *)(a1 + 195) & 0x80 | v32 & 0x7F;
    v5[12].m128i_i8[4] = *(_BYTE *)(a1 + 196) & 1 | v5[12].m128i_i8[4] & 0xFE;
    v33 = *(_BYTE *)(a1 + 194) & 0x40;
    v5[12].m128i_i8[2] = v33 | v30 & 0xBF;
    if ( (v33 & 0x40) != 0 )
      v5[14].m128i_i64[1] = *(_QWORD *)(a1 + 232);
    v34 = *(_QWORD *)(v5[9].m128i_i64[1] + 168);
    *(_QWORD *)(v34 + 40) = *(_QWORD *)(v12 + 40);
    *(_BYTE *)(v34 + 21) = *(_BYTE *)(v12 + 21) & 1 | *(_BYTE *)(v34 + 21) & 0xFE;
    sub_814560(v5, a1);
    if ( (*(_BYTE *)(a1 + 192) & 2) != 0 && ((a2 - 2) & 0xFD) != 0 )
    {
      v5[12].m128i_i8[0] |= 2u;
      v35 = *(_WORD *)(a1 + 224);
      if ( a2 != 1 )
      {
        v5[14].m128i_i16[0] = v35 + 1;
        *(_BYTE *)(v34 + 16) = *(_BYTE *)(v12 + 16) & 1 | *(_BYTE *)(v34 + 16) & 0xFE;
        goto LABEL_47;
      }
      v5[14].m128i_i16[0] = v35;
      *(_BYTE *)(v34 + 16) = *(_BYTE *)(v12 + 16) & 1 | *(_BYTE *)(v34 + 16) & 0xFE;
      if ( (*(_BYTE *)(*(_QWORD *)(*(_QWORD *)(a1 + 40) + 32LL) + 176LL) & 0x10) != 0 )
        goto LABEL_64;
    }
    else
    {
      *(_BYTE *)(v34 + 16) = *(_BYTE *)(v12 + 16) & 1 | *(_BYTE *)(v34 + 16) & 0xFE;
      if ( (*(_BYTE *)(*(_QWORD *)(*(_QWORD *)(a1 + 40) + 32LL) + 176LL) & 0x10) != 0 && a2 == 1 )
      {
LABEL_64:
        if ( *(_DWORD *)(a1 + 160)
          && *(_QWORD *)(sub_72B840(a1) + 48)
          && (unsigned __int8)(v5[10].m128i_i8[14] - 1) <= 1u )
        {
          *(_QWORD *)(v34 + 56) = *(_QWORD *)(v12 + 56);
        }
        goto LABEL_47;
      }
      if ( a2 == 2 && (*(_BYTE *)(a1 + 205) & 0x1C) == 4 )
        v5[12].m128i_i8[13] |= 0x20u;
    }
LABEL_47:
    v36 = *(_BYTE *)(a1 + 200) & 7 | v5[12].m128i_i8[8] & 0xF8;
    v5[12].m128i_i8[8] = v36;
    v37 = *(_BYTE *)(a1 + 200) & 0x20 | v36 & 0xDF;
    v5[12].m128i_i8[8] = v37;
    v5[12].m128i_i8[8] = *(_BYTE *)(a1 + 200) & 0x40 | v37 & 0xBF;
    v38 = *(_QWORD **)(a1 + 256);
    if ( v38 )
    {
      v39 = (_QWORD *)v5[16].m128i_i64[0];
      if ( !v39 )
        v39 = (_QWORD *)sub_726210((__int64)v5);
      *v39 = *v38;
    }
    v40 = sub_725220();
    v40[1] = v5;
    *v40 = *(_QWORD *)(a1 + 176);
    *(_QWORD *)(a1 + 176) = v40;
    v41 = v5[10].m128i_i8[14];
    v42 = *(_QWORD **)v34;
    if ( (v41 == 1 || v41 == 2)
      && (((v5[12].m128i_i8[13] & 0x1C) - 8) & 0xF4) == 0
      && (*(_BYTE *)(*(_QWORD *)(v5[2].m128i_i64[1] + 32) + 176LL) & 0x10) != 0 )
    {
      v49 = *(_QWORD **)v34;
      v43 = sub_7E1DF0();
      v44 = sub_724EF0(v43);
      *v49 = v44;
      v42 = v44;
    }
    sub_7F6570(a1, v42, 1, 0);
    v7 = *(_BYTE *)(a1 + 198);
    if ( (v7 & 8) == 0 )
      goto LABEL_10;
    goto LABEL_9;
  }
  return v5;
}
