// Function: sub_634B10
// Address: 0x634b10
//
__int64 __fastcall sub_634B10(__int64 *a1, __int64 a2, __int64 a3, __m128i *a4, __int64 a5, __int64 *a6)
{
  unsigned __int8 v9; // al
  __int64 v10; // r15
  __int64 v11; // rdi
  char v12; // bl
  __int64 v13; // rdx
  char i; // cl
  __int64 *v15; // rax
  __int64 v16; // rax
  char v17; // si
  int v18; // eax
  __int64 v19; // rdx
  char v20; // cl
  __int64 v21; // r8
  __int8 v22; // al
  __int64 v23; // rax
  unsigned __int64 v25; // rax
  __int64 v26; // rsi
  __int64 v27; // rsi
  __int64 v28; // rax
  __int64 v29; // rax
  __int64 k; // rsi
  int v31; // eax
  __int64 v32; // r8
  __int64 j; // rax
  __int64 v34; // rsi
  _QWORD *v35; // rax
  __int64 v36; // rax
  __int64 v37; // rdx
  __int64 v38; // rcx
  __int64 v39; // r9
  __int64 v40; // r11
  __int64 v41; // rax
  __int64 v42; // rdx
  __int64 v43; // rcx
  __int64 v44; // r8
  __int64 v45; // rax
  __int16 v46; // di
  __int64 v47; // rax
  __int64 v48; // r9
  __int64 v50; // [rsp+0h] [rbp-B0h]
  __int64 v51; // [rsp+0h] [rbp-B0h]
  bool v52; // [rsp+8h] [rbp-A8h]
  __int64 v53; // [rsp+8h] [rbp-A8h]
  __int64 v54; // [rsp+8h] [rbp-A8h]
  __int64 v55; // [rsp+8h] [rbp-A8h]
  char v56; // [rsp+10h] [rbp-A0h]
  __int64 v57; // [rsp+10h] [rbp-A0h]
  __int64 v58; // [rsp+10h] [rbp-A0h]
  __int64 v59; // [rsp+10h] [rbp-A0h]
  __int64 v60; // [rsp+10h] [rbp-A0h]
  __int64 v61; // [rsp+18h] [rbp-98h]
  char v62; // [rsp+18h] [rbp-98h]
  _QWORD *v63; // [rsp+18h] [rbp-98h]
  __int64 v65; // [rsp+20h] [rbp-90h]
  __int64 v66; // [rsp+20h] [rbp-90h]
  __int64 v67; // [rsp+28h] [rbp-88h]
  char v68; // [rsp+37h] [rbp-79h]
  __int64 v69; // [rsp+38h] [rbp-78h] BYREF
  _QWORD v70[14]; // [rsp+40h] [rbp-70h] BYREF

  v9 = a4[2].m128i_u8[9];
  v69 = a2;
  v10 = *a1;
  v68 = v9 >> 7;
  v11 = *a1;
  v67 = a4[2].m128i_i64[0];
  v52 = (a4[2].m128i_i8[11] & 4) != 0;
  a4[2].m128i_i8[11] &= ~4u;
  if ( (unsigned int)sub_6E1B40(v11) )
  {
    v12 = 1;
    v13 = *(_QWORD *)&dword_4D03B80;
    a4[2].m128i_i8[9] |= 0x20u;
    v69 = v13;
    i = *(_BYTE *)(v13 + 140);
  }
  else
  {
    v13 = v69;
    v12 = 0;
    for ( i = *(_BYTE *)(v69 + 140); i == 12; i = *(_BYTE *)(v13 + 140) )
      v13 = *(_QWORD *)(v13 + 160);
  }
  if ( !dword_4F077BC || (unsigned __int8)(i - 9) > 2u )
    goto LABEL_26;
  v15 = (__int64 *)a4[1].m128i_i64[0];
  if ( !v15 )
    goto LABEL_10;
  v16 = *v15;
  if ( !v16 )
    goto LABEL_10;
  v17 = *(_BYTE *)(v16 + 80);
  if ( v17 == 9 || v17 == 7 )
  {
    v23 = *(_QWORD *)(v16 + 88);
    goto LABEL_23;
  }
  if ( v17 == 21 )
  {
    v23 = *(_QWORD *)(*(_QWORD *)(v16 + 88) + 192LL);
LABEL_23:
    if ( !v23 || (*(_BYTE *)(v23 + 170) & 0x20) == 0 )
      goto LABEL_10;
    v13 = *(_QWORD *)&dword_4D03B80;
    v69 = *(_QWORD *)&dword_4D03B80;
    i = *(_BYTE *)(*(_QWORD *)&dword_4D03B80 + 140LL);
LABEL_26:
    if ( i == 8 )
    {
      a4[2].m128i_i8[9] |= 0x80u;
      a4[2].m128i_i64[0] = 0;
      sub_635980(a1, &v69, a4, a5, a6);
      v22 = a4[2].m128i_i8[8];
      goto LABEL_28;
    }
  }
LABEL_10:
  v56 = i;
  v61 = v13;
  v18 = sub_8D3BB0(v13);
  v19 = v61;
  v20 = v56;
  if ( !v18 )
    goto LABEL_11;
  if ( !dword_4D04428 )
    goto LABEL_50;
  if ( *(_BYTE *)(v10 + 8) != 1 )
  {
LABEL_36:
    v25 = a4[2].m128i_i64[1] & 0xFFFFFFFEFFFF7FFFLL;
    a4[2].m128i_i64[0] = 0;
    BYTE1(v25) |= 0x80u;
    v26 = v69;
    a4[2].m128i_i64[1] = v25;
    sub_6333F0(a1, v26, a4, a5, a6);
    v22 = a4[2].m128i_i8[8];
    goto LABEL_28;
  }
  if ( (unsigned __int8)(*(_BYTE *)(v61 + 140) - 9) > 2u
    || (v29 = *(_QWORD *)(v10 + 24)) == 0
    || *(_QWORD *)v29
    || *(_BYTE *)(v29 + 8) )
  {
    if ( !v52 )
      goto LABEL_36;
LABEL_51:
    if ( *(_QWORD *)(v10 + 24) || (*(_BYTE *)(*(_QWORD *)(*(_QWORD *)v19 + 96LL) + 176LL) & 1) == 0 )
      goto LABEL_36;
    v63 = (_QWORD *)v69;
    v32 = *(_QWORD *)(*(_QWORD *)(*(_QWORD *)(v69 + 168) + 152LL) + 144LL);
    if ( v32 )
    {
      while ( 1 )
      {
        if ( *(_BYTE *)(v32 + 174) == 1 && (*(_BYTE *)(v32 + 206) & 0x10) == 0 )
        {
          for ( j = *(_QWORD *)(v32 + 152); *(_BYTE *)(j + 140) == 12; j = *(_QWORD *)(j + 160) )
            ;
          if ( !**(_QWORD **)(j + 168) )
            break;
        }
        v32 = *(_QWORD *)(v32 + 112);
        if ( !v32 )
          goto LABEL_94;
      }
    }
    else
    {
LABEL_94:
      v51 = *(_QWORD *)(*(_QWORD *)(v69 + 168) + 152LL);
      v54 = sub_7259C0(7);
      v58 = *(_QWORD *)(v54 + 168);
      *(_QWORD *)(v54 + 160) = sub_72CBE0();
      v46 = *(_WORD *)(v58 + 16);
      *(_BYTE *)(v58 + 21) |= 1u;
      *(_QWORD *)(v58 + 40) = v63;
      *(_WORD *)(v58 + 16) = v46 & 0x8EFD | 0x2102;
      sub_7325D0(v54, &unk_4F077C8);
      sub_878710(*v63, v70);
      sub_87A680(v70, a5, 0);
      v59 = sub_87EBB0(10, v70[0]);
      *(_DWORD *)(v59 + 40) = *(_DWORD *)(v51 + 24);
      v47 = sub_646F50(v54, 2, 0xFFFFFFFFLL);
      *(_BYTE *)(v47 + 193) |= 0x10u;
      v48 = v59;
      v60 = v47;
      *(_QWORD *)(v48 + 88) = v47;
      v55 = v48;
      sub_877D80(v47, v48);
      sub_877E20(v55, v60, v63);
      sub_725ED0(v60, 1);
      sub_736C90(v60, 1);
      v32 = v60;
      *(_BYTE *)(v60 + 88) = *(_BYTE *)(v60 + 88) & 0x8F | 0x10;
      *(_QWORD *)(v60 + 112) = *(_QWORD *)(v51 + 144);
      *(_QWORD *)(v51 + 144) = v60;
      if ( dword_4F068EC )
      {
        sub_89A080(v60);
        v32 = v60;
      }
    }
    v53 = v32;
    sub_732AE0(v32);
    v57 = sub_630000((int)v63, a4, a5);
    if ( (a4[2].m128i_i8[8] & 0x40) != 0 )
    {
      *a6 = 0;
LABEL_38:
      v28 = *(_QWORD *)v10;
      if ( *(_QWORD *)v10 && *(_BYTE *)(v28 + 8) == 3 )
        v28 = sub_6BBB10(v10);
      *a1 = v28;
      v22 = a4[2].m128i_i8[8];
      goto LABEL_28;
    }
    v36 = sub_62FD00(v53, 0, (((unsigned __int8)a4[2].m128i_i8[10] >> 5) ^ 1) & 1, 0);
    *(_BYTE *)(v36 + 72) |= 4u;
    v40 = v36;
    if ( (*(_BYTE *)(v53 + 193) & 2) != 0 )
    {
      v50 = v36;
      v70[0] = sub_724DC0(v53, 0, v37, v38, v53, v39);
      if ( (unsigned int)sub_71AAF0(v50, 1, 0, (*(_BYTE *)(v53 + 193) & 4) != 0, a5, v70[0]) )
      {
        if ( (*(_BYTE *)(v70[0] + 170LL) & 0x40) != 0 )
          a4[2].m128i_i8[9] |= 0x10u;
        *a6 = sub_724E50(v70, 1, v42, v43, v44);
        if ( !v57 )
          goto LABEL_86;
        v40 = sub_725A70(2);
        v45 = *a6;
        *(_QWORD *)(v40 + 56) = *a6;
        if ( (*(_BYTE *)(v45 + 170) & 0x40) != 0 )
          *(_BYTE *)(v40 + 50) |= 0x80u;
      }
      else
      {
        sub_724E30(v70);
        v40 = v50;
      }
    }
    else
    {
      a4[2].m128i_i8[10] |= 0x80u;
    }
    if ( v57 )
    {
      *(_QWORD *)(v40 + 16) = v57;
      if ( (a4[2].m128i_i8[10] & 0x20) == 0 )
        *(_BYTE *)(v57 + 193) |= 0x40u;
      if ( (a4[2].m128i_i8[8] & 4) == 0 )
      {
        v66 = v40;
        sub_734250(v40, (((unsigned __int8)a4[2].m128i_i8[10] >> 4) ^ 1) & 1);
        v40 = v66;
      }
    }
    v65 = v40;
    v41 = sub_724D50(9);
    *a6 = v41;
    *(_QWORD *)(v41 + 176) = v65;
    *(_QWORD *)(*a6 + 128) = v63;
    a4[2].m128i_i8[9] |= 4u;
LABEL_86:
    *(_QWORD *)(*a6 + 64) = *(_QWORD *)sub_6E1A20(v10);
    if ( *(_BYTE *)(v10 + 8) != 2 )
      *(_QWORD *)(*a6 + 112) = *(_QWORD *)sub_6E1A60(v10);
    goto LABEL_38;
  }
  for ( k = *(_QWORD *)(*(_QWORD *)(v29 + 24) + 8LL); *(_BYTE *)(k + 140) == 12; k = *(_QWORD *)(k + 160) )
    ;
  v31 = sub_8DF8D0(v61, k);
  v19 = v61;
  v20 = v56;
  if ( !v31 )
  {
LABEL_50:
    if ( !v52 || *(_BYTE *)(v10 + 8) != 1 )
      goto LABEL_36;
    goto LABEL_51;
  }
LABEL_11:
  v62 = v20;
  if ( (unsigned int)sub_8DD3B0(v69) || !v62 )
  {
    a4[2].m128i_i8[9] |= 0x80u;
    v27 = v69;
    a4[2].m128i_i64[0] = 0;
    sub_6319F0((_QWORD *)v10, v27, a4, a6);
    goto LABEL_38;
  }
  if ( v62 == 15 )
  {
    if ( dword_4F077BC || dword_4F077C0 && !(_DWORD)qword_4F077B4 && qword_4F077A8 > 0x9E33u || *(_BYTE *)(v10 + 8) == 1 )
    {
      a4[2].m128i_i8[9] |= 0x80u;
      v34 = v69;
      a4[2].m128i_i64[0] = 0;
      sub_636A00(a1, v34, a4, a5, a6);
      v22 = a4[2].m128i_i8[8];
      goto LABEL_28;
    }
  }
  else if ( (dword_4F077BC && qword_4F077A8 > 0x9EFBu || (_DWORD)qword_4F077B4) && v62 == 5 && *(_BYTE *)(v10 + 8) == 1 )
  {
    v35 = *(_QWORD **)(v10 + 24);
    if ( v35 )
    {
      if ( *v35 )
      {
        sub_636FC0(a1, v69, a4, a6);
        v22 = a4[2].m128i_i8[8];
        goto LABEL_28;
      }
    }
  }
  sub_631120(a1, v69, a4, (__int64)a6);
  if ( !*a6 )
  {
LABEL_21:
    v22 = a4[2].m128i_i8[8];
    goto LABEL_28;
  }
  v21 = sub_6E1A20(v10);
  v22 = a4[2].m128i_i8[8];
  if ( (v22 & 6) == 6 && *(_BYTE *)(*a6 + 173) == 6 )
  {
    sub_630E60(*(_QWORD *)(*a6 + 128), *(_QWORD *)(*a6 + 136), v69, a3, v21);
    goto LABEL_21;
  }
LABEL_28:
  if ( (v22 & 0x40) == 0 )
    *(_BYTE *)(*a6 + 170) = (4 * (v12 & 1)) | *(_BYTE *)(*a6 + 170) & 0xFB;
  a4[2].m128i_i8[9] = (v68 << 7) | a4[2].m128i_i8[9] & 0x7F;
  a4[2].m128i_i64[0] = v67;
  return v67;
}
