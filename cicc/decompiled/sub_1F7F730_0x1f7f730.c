// Function: sub_1F7F730
// Address: 0x1f7f730
//
__int64 __fastcall sub_1F7F730(
        __int64 a1,
        __int64 a2,
        __int64 *a3,
        char a4,
        char a5,
        __m128i a6,
        double a7,
        __m128i a8)
{
  char *v10; // rdx
  __int64 *v11; // rax
  const void **v12; // rcx
  __int64 v13; // r13
  __int64 v14; // r10
  __int64 v15; // r11
  unsigned __int8 v16; // al
  int v17; // edx
  __int64 v18; // rdx
  __int64 v19; // r12
  __int64 v21; // rsi
  bool v22; // al
  bool v23; // al
  __int64 v24; // rcx
  __int64 v25; // r8
  __int64 v26; // r9
  unsigned __int8 *v27; // rax
  __int64 v28; // rdx
  __int64 v29; // rax
  __int64 v30; // r9
  __int64 v31; // r15
  __int64 v32; // rsi
  __int64 v33; // rbx
  __int64 v34; // r8
  __int64 v35; // rdx
  __int64 v36; // rdx
  __int64 *v37; // rdx
  __int64 v38; // rcx
  __int64 v39; // rsi
  __int64 v40; // rsi
  unsigned __int8 v41; // al
  const void **v42; // rdx
  const void **v43; // r9
  unsigned __int8 v44; // cl
  _QWORD *v45; // rax
  __int64 v46; // rdx
  _QWORD *v47; // r8
  __int64 v48; // rdx
  _QWORD *v49; // rdx
  __int128 v50; // [rsp-10h] [rbp-190h]
  __int128 v51; // [rsp-10h] [rbp-190h]
  __int64 v52; // [rsp+20h] [rbp-160h]
  _QWORD *v53; // [rsp+20h] [rbp-160h]
  __int64 v54; // [rsp+20h] [rbp-160h]
  _QWORD *v55; // [rsp+20h] [rbp-160h]
  __int64 v56; // [rsp+28h] [rbp-158h]
  __int64 v57; // [rsp+28h] [rbp-158h]
  __int64 v58; // [rsp+28h] [rbp-158h]
  char v59; // [rsp+34h] [rbp-14Ch]
  char v60; // [rsp+38h] [rbp-148h]
  unsigned int v61; // [rsp+38h] [rbp-148h]
  __int64 v62; // [rsp+40h] [rbp-140h]
  unsigned __int8 v63; // [rsp+40h] [rbp-140h]
  __int64 v64; // [rsp+40h] [rbp-140h]
  __int64 v65; // [rsp+48h] [rbp-138h]
  const void **v66; // [rsp+50h] [rbp-130h]
  unsigned int v67; // [rsp+50h] [rbp-130h]
  unsigned __int16 v68; // [rsp+58h] [rbp-128h]
  __int64 v69; // [rsp+60h] [rbp-120h] BYREF
  const void **v70; // [rsp+68h] [rbp-118h]
  __int64 v71; // [rsp+70h] [rbp-110h] BYREF
  const void **v72; // [rsp+78h] [rbp-108h]
  __int64 v73; // [rsp+80h] [rbp-100h] BYREF
  int v74; // [rsp+88h] [rbp-F8h]
  __int64 v75; // [rsp+90h] [rbp-F0h] BYREF
  int v76; // [rsp+98h] [rbp-E8h]
  const void *v77; // [rsp+A0h] [rbp-E0h] BYREF
  unsigned int v78; // [rsp+A8h] [rbp-D8h]
  __int64 v79; // [rsp+B0h] [rbp-D0h] BYREF
  unsigned int v80; // [rsp+B8h] [rbp-C8h]
  _BYTE *v81; // [rsp+C0h] [rbp-C0h] BYREF
  __int64 v82; // [rsp+C8h] [rbp-B8h]
  _BYTE v83[176]; // [rsp+D0h] [rbp-B0h] BYREF

  v10 = *(char **)(a1 + 40);
  v68 = *(_WORD *)(a1 + 24);
  v11 = *(__int64 **)(a1 + 32);
  v12 = (const void **)*((_QWORD *)v10 + 1);
  v13 = *v11;
  v14 = *v11;
  v15 = v11[1];
  v16 = *v10;
  v70 = v12;
  v17 = *(unsigned __int16 *)(v13 + 24);
  LOBYTE(v69) = v16;
  if ( v17 == 10 || v17 == 32 )
  {
    v21 = *(_QWORD *)(a1 + 72);
    v81 = (_BYTE *)v21;
    if ( v21 )
    {
      v62 = v14;
      v65 = v15;
      sub_1623A60((__int64)&v81, v21, 2);
      v14 = v62;
      v15 = v65;
    }
    *((_QWORD *)&v50 + 1) = v15;
    *(_QWORD *)&v50 = v14;
    LODWORD(v82) = *(_DWORD *)(a1 + 64);
    v19 = sub_1D309E0(
            a3,
            v68,
            (__int64)&v81,
            (unsigned int)v69,
            v70,
            0,
            *(double *)a6.m128i_i64,
            a7,
            *(double *)a8.m128i_i64,
            v50);
    if ( v81 )
      sub_161E7C0((__int64)&v81, (__int64)v81);
    return v19;
  }
  if ( v16 )
  {
    if ( (unsigned __int8)(v16 - 14) <= 0x5Fu )
    {
      switch ( v16 )
      {
        case 0x18u:
        case 0x19u:
        case 0x1Au:
        case 0x1Bu:
        case 0x1Cu:
        case 0x1Du:
        case 0x1Eu:
        case 0x1Fu:
        case 0x20u:
        case 0x3Eu:
        case 0x3Fu:
        case 0x40u:
        case 0x41u:
        case 0x42u:
        case 0x43u:
          v18 = 3;
          break;
        case 0x21u:
        case 0x22u:
        case 0x23u:
        case 0x24u:
        case 0x25u:
        case 0x26u:
        case 0x27u:
        case 0x28u:
        case 0x44u:
        case 0x45u:
        case 0x46u:
        case 0x47u:
        case 0x48u:
        case 0x49u:
          v18 = 4;
          break;
        case 0x29u:
        case 0x2Au:
        case 0x2Bu:
        case 0x2Cu:
        case 0x2Du:
        case 0x2Eu:
        case 0x2Fu:
        case 0x30u:
        case 0x4Au:
        case 0x4Bu:
        case 0x4Cu:
        case 0x4Du:
        case 0x4Eu:
        case 0x4Fu:
          v18 = 5;
          break;
        case 0x31u:
        case 0x32u:
        case 0x33u:
        case 0x34u:
        case 0x35u:
        case 0x36u:
        case 0x50u:
        case 0x51u:
        case 0x52u:
        case 0x53u:
        case 0x54u:
        case 0x55u:
          v18 = 6;
          break;
        case 0x37u:
          v18 = 7;
          break;
        case 0x56u:
        case 0x57u:
        case 0x58u:
        case 0x62u:
        case 0x63u:
        case 0x64u:
          v18 = 8;
          break;
        case 0x59u:
        case 0x5Au:
        case 0x5Bu:
        case 0x5Cu:
        case 0x5Du:
        case 0x65u:
        case 0x66u:
        case 0x67u:
        case 0x68u:
        case 0x69u:
          v18 = 9;
          break;
        case 0x5Eu:
        case 0x5Fu:
        case 0x60u:
        case 0x61u:
        case 0x6Au:
        case 0x6Bu:
        case 0x6Cu:
        case 0x6Du:
          v18 = 10;
          break;
        default:
          v18 = 2;
          break;
      }
      LOBYTE(v71) = v18;
      v72 = 0;
    }
    else
    {
      LOBYTE(v71) = v16;
      v18 = v16;
      v72 = v12;
    }
LABEL_6:
    if ( (unsigned __int8)(v16 - 14) > 0x5Fu )
      return 0;
    goto LABEL_16;
  }
  v59 = a5;
  v66 = v12;
  v22 = sub_1F58D20((__int64)&v69);
  a5 = v59;
  if ( v22 )
  {
    v41 = sub_1F596B0((__int64)&v69);
    v43 = v42;
    v44 = v41;
    v18 = v41;
    v16 = v69;
    LOBYTE(v71) = v44;
    a5 = v59;
    v72 = v43;
    if ( (_BYTE)v69 )
      goto LABEL_6;
  }
  else
  {
    LOBYTE(v71) = 0;
    LOBYTE(v18) = 0;
    v72 = v66;
  }
  v60 = a5;
  v63 = v18;
  v23 = sub_1F58D20((__int64)&v69);
  v18 = v63;
  a5 = v60;
  if ( !v23 )
    return 0;
LABEL_16:
  if ( a4 && (a5 || !(_BYTE)v18 || !*(_QWORD *)(a2 + 8 * v18 + 120)) || !(unsigned __int8)sub_1D168E0(v13) )
    return 0;
  if ( (_BYTE)v71 )
    v61 = sub_1F6C8D0(v71);
  else
    v61 = sub_1F58D40((__int64)&v71);
  v27 = *(unsigned __int8 **)(v13 + 40);
  v28 = *v27;
  v29 = *((_QWORD *)v27 + 1);
  LOBYTE(v81) = v28;
  v82 = v29;
  v67 = sub_1D159C0((__int64)&v81, a2, v28, v24, v25, v26);
  v81 = v83;
  v82 = 0x800000000LL;
  if ( (_BYTE)v69 )
    v31 = word_42FA680[(unsigned __int8)(v69 - 14)];
  else
    v31 = (unsigned int)sub_1F58D30((__int64)&v69);
  v32 = *(_QWORD *)(a1 + 72);
  v73 = v32;
  if ( v32 )
    sub_1623A60((__int64)&v73, v32, 2);
  v74 = *(_DWORD *)(a1 + 64);
  if ( (_DWORD)v31 )
  {
    v33 = 0;
    v64 = 40 * v31;
    do
    {
      while ( 1 )
      {
        v38 = *(_QWORD *)(*(_QWORD *)(v13 + 32) + v33);
        if ( *(_WORD *)(v38 + 24) == 48 )
          break;
        v39 = *(_QWORD *)(v38 + 72);
        v75 = v39;
        if ( v39 )
        {
          v52 = v38;
          sub_1623A60((__int64)&v75, v39, 2);
          v38 = v52;
        }
        v40 = *(_QWORD *)(v38 + 88) + 24LL;
        v76 = *(_DWORD *)(v38 + 64);
        sub_16A5D10((__int64)&v77, v40, v67);
        if ( ((v68 - 142) & 0xFFF7) != 0 )
          sub_16A5C50((__int64)&v79, &v77, v61);
        else
          sub_16A5B10((__int64)&v79, &v77, v61);
        v34 = sub_1D38970((__int64)a3, (__int64)&v79, (__int64)&v75, v71, v72, 0, a6, a7, a8, 0);
        v30 = v35;
        v36 = (unsigned int)v82;
        if ( (unsigned int)v82 >= HIDWORD(v82) )
        {
          v54 = v34;
          v57 = v30;
          sub_16CD150((__int64)&v81, v83, 0, 16, v34, v30);
          v36 = (unsigned int)v82;
          v34 = v54;
          v30 = v57;
        }
        v37 = (__int64 *)&v81[16 * v36];
        *v37 = v34;
        v37[1] = v30;
        LODWORD(v82) = v82 + 1;
        if ( v80 > 0x40 && v79 )
          j_j___libc_free_0_0(v79);
        if ( v78 > 0x40 && v77 )
          j_j___libc_free_0_0(v77);
        if ( v75 )
          sub_161E7C0((__int64)&v75, v75);
        v33 += 40;
        if ( v33 == v64 )
          goto LABEL_54;
      }
      v79 = 0;
      v80 = 0;
      v45 = sub_1D2B300(a3, 0x30u, (__int64)&v79, v71, (__int64)v72, v30);
      v47 = v45;
      v30 = v46;
      if ( v79 )
      {
        v53 = v45;
        v56 = v46;
        sub_161E7C0((__int64)&v79, v79);
        v47 = v53;
        v30 = v56;
      }
      v48 = (unsigned int)v82;
      if ( (unsigned int)v82 >= HIDWORD(v82) )
      {
        v55 = v47;
        v58 = v30;
        sub_16CD150((__int64)&v81, v83, 0, 16, (int)v47, v30);
        v48 = (unsigned int)v82;
        v47 = v55;
        v30 = v58;
      }
      v49 = &v81[16 * v48];
      v33 += 40;
      *v49 = v47;
      v49[1] = v30;
      LODWORD(v82) = v82 + 1;
    }
    while ( v33 != v64 );
  }
LABEL_54:
  *((_QWORD *)&v51 + 1) = (unsigned int)v82;
  *(_QWORD *)&v51 = v81;
  v19 = (__int64)sub_1D359D0(a3, 104, (__int64)&v73, v69, v70, 0, *(double *)a6.m128i_i64, a7, a8, v51);
  if ( v73 )
    sub_161E7C0((__int64)&v73, v73);
  if ( v81 != v83 )
    _libc_free((unsigned __int64)v81);
  return v19;
}
