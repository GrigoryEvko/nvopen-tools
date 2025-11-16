// Function: sub_1D3B650
// Address: 0x1d3b650
//
__int64 *__fastcall sub_1D3B650(
        __int64 *a1,
        unsigned int a2,
        const void **a3,
        __int64 a4,
        __int64 a5,
        __int64 a6,
        __m128 a7,
        __m128i a8,
        __m128i a9,
        unsigned __int64 a10,
        __int16 *a11,
        __int64 a12)
{
  unsigned __int8 *v16; // rax
  __int64 v17; // rcx
  __int64 v18; // rdx
  const void **v19; // r8
  __int64 v20; // rdx
  __int64 v21; // rsi
  __int64 v22; // rcx
  int v24; // edi
  int v25; // eax
  __int64 v26; // rdi
  __int64 v27; // r11
  unsigned int v28; // eax
  __int64 v29; // rdx
  __int64 v30; // rax
  __int64 v31; // rdx
  unsigned int v32; // eax
  __int64 v33; // r8
  int v34; // eax
  unsigned int v35; // eax
  int v36; // eax
  int v37; // eax
  bool v38; // al
  bool v39; // al
  bool v40; // si
  bool v41; // zf
  __int128 v42; // [rsp-20h] [rbp-80h]
  __int64 v43; // [rsp+8h] [rbp-58h]
  __int64 v45; // [rsp+10h] [rbp-50h]
  __int64 v46; // [rsp+10h] [rbp-50h]
  __int64 v47; // [rsp+10h] [rbp-50h]
  __int64 v48; // [rsp+10h] [rbp-50h]
  __int64 v49; // [rsp+10h] [rbp-50h]
  __int64 v50; // [rsp+10h] [rbp-50h]
  __int64 v51; // [rsp+10h] [rbp-50h]
  unsigned __int16 v52; // [rsp+20h] [rbp-40h]
  __int64 v53; // [rsp+20h] [rbp-40h]
  unsigned int v54; // [rsp+20h] [rbp-40h]
  __int64 v55; // [rsp+20h] [rbp-40h]
  __int64 v56; // [rsp+20h] [rbp-40h]
  __int64 v57; // [rsp+20h] [rbp-40h]
  __int64 v58; // [rsp+20h] [rbp-40h]
  __int64 v59; // [rsp+20h] [rbp-40h]
  __int64 v60; // [rsp+20h] [rbp-40h]
  __int128 v61; // [rsp+70h] [rbp+10h]

  v16 = (unsigned __int8 *)(16LL * (unsigned int)a5 + *(_QWORD *)(a4 + 40));
  v17 = *((_QWORD *)v16 + 1);
  v18 = *v16;
  if ( (_DWORD)a6 == 16 )
    goto LABEL_13;
  if ( (unsigned int)a6 > 0x10 )
  {
    if ( (_DWORD)a6 == 23 )
      goto LABEL_5;
    goto LABEL_8;
  }
  if ( !(_DWORD)a6 )
  {
LABEL_13:
    *(_QWORD *)&v61 = *v16;
    v19 = a3;
    v20 = a12;
    v21 = 0;
    *((_QWORD *)&v61 + 1) = *((_QWORD *)v16 + 1);
    v22 = a2;
    return (__int64 *)sub_1D395A0((__int64)a1, v21, v20, v22, v19, a6, *(double *)a7.m128_u64, a8, a9, v61);
  }
  if ( (_DWORD)a6 == 15 )
  {
LABEL_5:
    *(_QWORD *)&v61 = *v16;
    v19 = a3;
    v20 = a12;
    v21 = 1;
    *((_QWORD *)&v61 + 1) = *((_QWORD *)v16 + 1);
    v22 = a2;
    return (__int64 *)sub_1D395A0((__int64)a1, v21, v20, v22, v19, a6, *(double *)a7.m128_u64, a8, a9, v61);
  }
LABEL_8:
  v24 = *(unsigned __int16 *)(a4 + 24);
  v25 = *(unsigned __int16 *)(a10 + 24);
  v52 = *(_WORD *)(a4 + 24);
  if ( v25 == 32 || v25 == 10 )
  {
    v24 = v52;
    if ( v52 == 10 || v52 == 32 )
    {
      v26 = *(_QWORD *)(a10 + 88);
      v27 = *(_QWORD *)(a4 + 88);
      switch ( (int)a6 )
      {
        case 10:
          v49 = v18;
          v58 = v17;
          v37 = sub_16A9900(v27 + 24, (unsigned __int64 *)(v26 + 24));
          goto LABEL_30;
        case 11:
          v48 = v18;
          v57 = v17;
          v36 = sub_16A9900(v27 + 24, (unsigned __int64 *)(v26 + 24));
          goto LABEL_28;
        case 12:
          v47 = v18;
          v56 = v17;
          v35 = sub_16A9900(v27 + 24, (unsigned __int64 *)(v26 + 24));
          goto LABEL_26;
        case 13:
          v46 = v18;
          v55 = v17;
          v34 = sub_16A9900(v27 + 24, (unsigned __int64 *)(v26 + 24));
          goto LABEL_23;
        case 14:
        case 22:
          if ( *(_DWORD *)(v27 + 32) <= 0x40u )
          {
            v40 = *(_QWORD *)(v27 + 24) == *(_QWORD *)(v26 + 24);
          }
          else
          {
            v51 = v18;
            v60 = v17;
            v39 = sub_16A5220(v27 + 24, (const void **)(v26 + 24));
            v18 = v51;
            v17 = v60;
            v40 = v39;
          }
          LOBYTE(v21) = !v40;
          goto LABEL_42;
        case 17:
          if ( *(_DWORD *)(v27 + 32) <= 0x40u )
          {
            v38 = *(_QWORD *)(v27 + 24) == *(_QWORD *)(v26 + 24);
          }
          else
          {
            v50 = v18;
            v59 = v17;
            v38 = sub_16A5220(v27 + 24, (const void **)(v26 + 24));
            v18 = v50;
            v17 = v59;
          }
          v21 = v38;
          goto LABEL_34;
        case 18:
          v49 = v18;
          v58 = v17;
          v37 = sub_16AEA10(v27 + 24, v26 + 24);
LABEL_30:
          v17 = v58;
          v21 = v37 > 0;
          *(_QWORD *)&v61 = v49;
          break;
        case 19:
          v48 = v18;
          v57 = v17;
          v36 = sub_16AEA10(v27 + 24, v26 + 24);
LABEL_28:
          v17 = v57;
          *(_QWORD *)&v61 = v48;
          v21 = v36 >= 0;
          break;
        case 20:
          v47 = v18;
          v56 = v17;
          v35 = sub_16AEA10(v27 + 24, v26 + 24);
LABEL_26:
          v17 = v56;
          v21 = v35 >> 31;
          *(_QWORD *)&v61 = v47;
          break;
        case 21:
          v46 = v18;
          v55 = v17;
          v34 = sub_16AEA10(v27 + 24, v26 + 24);
LABEL_23:
          v17 = v55;
          v21 = v34 <= 0;
          *(_QWORD *)&v61 = v46;
          break;
      }
LABEL_24:
      *((_QWORD *)&v61 + 1) = v17;
      v19 = a3;
      v22 = a2;
      v20 = a12;
      return (__int64 *)sub_1D395A0((__int64)a1, v21, v20, v22, v19, a6, *(double *)a7.m128_u64, a8, a9, v61);
    }
  }
  if ( v24 != 11 && v24 != 33 )
    return 0;
  if ( v25 == 33 || v25 == 11 )
  {
    v54 = a6;
    v43 = v18;
    v45 = v17;
    v32 = sub_14A9E40(*(_QWORD *)(a4 + 88) + 24LL, *(_QWORD *)(a10 + 88) + 24LL);
    a6 = v54;
    v17 = v45;
    v18 = v43;
    switch ( v54 )
    {
      case 1u:
        goto LABEL_46;
      case 2u:
        goto LABEL_48;
      case 3u:
        goto LABEL_52;
      case 4u:
        goto LABEL_50;
      case 5u:
        goto LABEL_44;
      case 6u:
        goto LABEL_53;
      case 7u:
        v41 = v32 == 3;
        goto LABEL_56;
      case 8u:
        LOBYTE(v21) = v32 == 3;
        goto LABEL_42;
      case 9u:
        v32 &= ~2u;
        goto LABEL_46;
      case 0xAu:
        v32 -= 2;
        goto LABEL_44;
      case 0xBu:
        v41 = v32 == 0;
        goto LABEL_56;
      case 0xCu:
        LOBYTE(v21) = v32 == 0 || v32 == 3;
        goto LABEL_42;
      case 0xDu:
        v41 = v32 == 2;
        goto LABEL_56;
      case 0xEu:
        v41 = v32 == 1;
LABEL_56:
        LOBYTE(v21) = !v41;
        goto LABEL_42;
      case 0x11u:
        if ( v32 == 3 )
          return sub_1D2B530(a1, a2, (__int64)a3, v45, v33, v54);
LABEL_46:
        LOBYTE(v21) = v32 == 1;
        goto LABEL_42;
      case 0x12u:
        if ( v32 == 3 )
          return sub_1D2B530(a1, a2, (__int64)a3, v45, v33, v54);
LABEL_48:
        LOBYTE(v21) = v32 == 2;
        goto LABEL_42;
      case 0x13u:
        if ( v32 == 3 )
          return sub_1D2B530(a1, a2, (__int64)a3, v45, v33, v54);
LABEL_52:
        LOBYTE(v21) = v32 - 1 <= 1;
        goto LABEL_42;
      case 0x14u:
        if ( v32 == 3 )
          return sub_1D2B530(a1, a2, (__int64)a3, v45, v33, v54);
LABEL_50:
        LOBYTE(v21) = v32 == 0;
        goto LABEL_42;
      case 0x15u:
        if ( v32 == 3 )
          return sub_1D2B530(a1, a2, (__int64)a3, v45, v33, v54);
LABEL_44:
        LOBYTE(v21) = v32 <= 1;
        goto LABEL_42;
      case 0x16u:
        if ( v32 == 3 )
          return sub_1D2B530(a1, a2, (__int64)a3, v45, v33, v54);
LABEL_53:
        LOBYTE(v21) = (v32 & 0xFFFFFFFD) == 0;
LABEL_42:
        v21 = (unsigned __int8)v21;
LABEL_34:
        *(_QWORD *)&v61 = v18;
        break;
      default:
        return 0;
    }
    goto LABEL_24;
  }
  v53 = 16LL * (unsigned int)a5;
  v28 = sub_1D16ED0(a6);
  v29 = a1[2];
  if ( ((*(_DWORD *)(v29 + 4 * (((*(_BYTE *)(*(_QWORD *)(a4 + 40) + v53) >> 3) & 0x1F) + 15LL * (int)v28 + 18112) + 12) >> (4 * (*(_BYTE *)(*(_QWORD *)(a4 + 40) + v53) & 7)))
      & 0xF) != 0 )
    return 0;
  v30 = sub_1D28D50(a1, v28, v29, 4 * (*(_BYTE *)(*(_QWORD *)(a4 + 40) + v53) & 7u), a5, v28);
  *((_QWORD *)&v42 + 1) = a5;
  *(_QWORD *)&v42 = a4;
  return sub_1D3A900(a1, 0x89u, a12, a2, a3, 0, a7, *(double *)a8.m128i_i64, a9, a10, a11, v42, v30, v31);
}
