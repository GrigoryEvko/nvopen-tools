// Function: sub_20B8A00
// Address: 0x20b8a00
//
__int64 *__fastcall sub_20B8A00(__m128i a1, __int64 a2, __int64 a3, __int64 *a4)
{
  __int64 v5; // rbx
  __int64 v6; // rsi
  char v7; // r15
  __int64 v8; // rdx
  __int64 v9; // rax
  __int128 v10; // xmm1
  __m128i v11; // xmm2
  __int64 *v12; // rcx
  unsigned __int8 v13; // r15
  bool v14; // al
  __int64 v15; // rdx
  __int64 v16; // rax
  unsigned __int8 v17; // cl
  __int64 v18; // rax
  unsigned int v19; // eax
  __int64 *v20; // r8
  unsigned int v21; // esi
  const void **v22; // r15
  unsigned __int64 v23; // r15
  unsigned int v24; // ebx
  int v25; // edx
  unsigned __int64 v26; // r15
  __int128 v27; // rax
  __int64 *v28; // rax
  unsigned int v29; // edx
  __int64 v30; // r10
  unsigned int v31; // r9d
  __int64 v32; // rax
  __int64 *v33; // rax
  __int64 v34; // rax
  __int64 *v35; // rax
  __int64 v36; // rax
  __int64 v37; // rdi
  unsigned int v38; // eax
  __int64 v39; // rsi
  __int64 v40; // rax
  unsigned __int64 v41; // r9
  __int64 v42; // rdi
  char v43; // r10
  __int64 v44; // rdx
  bool v45; // zf
  _BYTE *v46; // rdx
  __int64 *v47; // r14
  __int64 v48; // rdx
  __int64 v49; // r15
  __int64 *v50; // rax
  __int32 v51; // edx
  int v52; // r8d
  __int64 *v53; // r14
  bool v55; // al
  char v56; // al
  __int64 v57; // rdx
  __int128 v58; // [rsp-20h] [rbp-270h]
  __int128 v59; // [rsp-10h] [rbp-260h]
  __int64 v60; // [rsp+10h] [rbp-240h]
  unsigned int v61; // [rsp+18h] [rbp-238h]
  unsigned int v62; // [rsp+34h] [rbp-21Ch]
  __int64 v63; // [rsp+38h] [rbp-218h]
  unsigned int v64; // [rsp+48h] [rbp-208h]
  int v65; // [rsp+4Ch] [rbp-204h]
  unsigned __int16 v66; // [rsp+50h] [rbp-200h]
  __int64 *v67; // [rsp+50h] [rbp-200h]
  __int64 *v68; // [rsp+50h] [rbp-200h]
  __int64 v69; // [rsp+60h] [rbp-1F0h]
  unsigned int v70; // [rsp+60h] [rbp-1F0h]
  unsigned int v71; // [rsp+60h] [rbp-1F0h]
  __int64 v72; // [rsp+68h] [rbp-1E8h]
  __int64 v73; // [rsp+68h] [rbp-1E8h]
  __int64 v74; // [rsp+68h] [rbp-1E8h]
  const void **v75; // [rsp+70h] [rbp-1E0h]
  __int64 v76; // [rsp+78h] [rbp-1D8h]
  unsigned int v77; // [rsp+78h] [rbp-1D8h]
  unsigned int v78; // [rsp+80h] [rbp-1D0h]
  __int64 v79; // [rsp+80h] [rbp-1D0h]
  __int64 *v80; // [rsp+88h] [rbp-1C8h]
  int v81; // [rsp+88h] [rbp-1C8h]
  __int64 v82; // [rsp+90h] [rbp-1C0h] BYREF
  int v83; // [rsp+98h] [rbp-1B8h]
  char v84[8]; // [rsp+A0h] [rbp-1B0h] BYREF
  __int64 v85; // [rsp+A8h] [rbp-1A8h]
  __int64 v86; // [rsp+B0h] [rbp-1A0h] BYREF
  __int64 v87; // [rsp+B8h] [rbp-198h]
  __int128 v88; // [rsp+C0h] [rbp-190h]
  __int64 v89; // [rsp+D0h] [rbp-180h]
  __m128i v90; // [rsp+E0h] [rbp-170h] BYREF
  __int64 *v91; // [rsp+F0h] [rbp-160h]
  __int64 v92; // [rsp+F8h] [rbp-158h]
  _BYTE *v93; // [rsp+100h] [rbp-150h] BYREF
  __int64 v94; // [rsp+108h] [rbp-148h]
  _BYTE v95[128]; // [rsp+110h] [rbp-140h] BYREF
  _BYTE *v96; // [rsp+190h] [rbp-C0h] BYREF
  __int64 v97; // [rsp+198h] [rbp-B8h]
  _BYTE v98[176]; // [rsp+1A0h] [rbp-B0h] BYREF

  v5 = a3;
  v6 = *(_QWORD *)(a3 + 72);
  v82 = v6;
  if ( v6 )
    sub_1623A60((__int64)&v82, v6, 2);
  v7 = *(_BYTE *)(v5 + 88);
  v8 = *(_QWORD *)(v5 + 96);
  v83 = *(_DWORD *)(v5 + 64);
  v9 = *(_QWORD *)(v5 + 32);
  v10 = (__int128)_mm_loadu_si128((const __m128i *)v9);
  v11 = _mm_loadu_si128((const __m128i *)(v9 + 40));
  v84[0] = v7;
  v12 = *(__int64 **)(v9 + 40);
  LODWORD(v9) = *(_DWORD *)(v9 + 48);
  v85 = v8;
  v78 = v9;
  v80 = v12;
  v62 = (*(_BYTE *)(v5 + 27) >> 2) & 3;
  if ( v7 )
  {
    v13 = v7 - 14;
    v65 = word_430A1A0[v13];
    switch ( v13 )
    {
      case 0u:
      case 1u:
      case 2u:
      case 3u:
      case 4u:
      case 5u:
      case 6u:
      case 7u:
      case 8u:
      case 9u:
      case 0x2Au:
      case 0x2Bu:
      case 0x2Cu:
      case 0x2Du:
      case 0x2Eu:
      case 0x2Fu:
        v7 = 2;
        v15 = 0;
        break;
      case 0xAu:
      case 0xBu:
      case 0xCu:
      case 0xDu:
      case 0xEu:
      case 0xFu:
      case 0x10u:
      case 0x11u:
      case 0x12u:
      case 0x30u:
      case 0x31u:
      case 0x32u:
      case 0x33u:
      case 0x34u:
      case 0x35u:
        v7 = 3;
        v15 = 0;
        break;
      case 0x13u:
      case 0x14u:
      case 0x15u:
      case 0x16u:
      case 0x17u:
      case 0x18u:
      case 0x19u:
      case 0x1Au:
      case 0x36u:
      case 0x37u:
      case 0x38u:
      case 0x39u:
      case 0x3Au:
      case 0x3Bu:
        v7 = 4;
        v15 = 0;
        break;
      case 0x1Bu:
      case 0x1Cu:
      case 0x1Du:
      case 0x1Eu:
      case 0x1Fu:
      case 0x20u:
      case 0x21u:
      case 0x22u:
      case 0x3Cu:
      case 0x3Du:
      case 0x3Eu:
      case 0x3Fu:
      case 0x40u:
      case 0x41u:
        v7 = 5;
        v15 = 0;
        break;
      case 0x23u:
      case 0x24u:
      case 0x25u:
      case 0x26u:
      case 0x27u:
      case 0x28u:
      case 0x42u:
      case 0x43u:
      case 0x44u:
      case 0x45u:
      case 0x46u:
      case 0x47u:
        v7 = 6;
        v15 = 0;
        break;
      case 0x29u:
        v7 = 7;
        v15 = 0;
        break;
      case 0x48u:
      case 0x49u:
      case 0x4Au:
      case 0x54u:
      case 0x55u:
      case 0x56u:
        v7 = 8;
        v15 = 0;
        break;
      case 0x4Bu:
      case 0x4Cu:
      case 0x4Du:
      case 0x4Eu:
      case 0x4Fu:
      case 0x57u:
      case 0x58u:
      case 0x59u:
      case 0x5Au:
      case 0x5Bu:
        v7 = 9;
        v15 = 0;
        break;
      case 0x50u:
      case 0x51u:
      case 0x52u:
      case 0x53u:
      case 0x5Cu:
      case 0x5Du:
      case 0x5Eu:
      case 0x5Fu:
        v7 = 10;
        v15 = 0;
        break;
    }
  }
  else
  {
    v76 = v8;
    v65 = sub_1F58D30((__int64)v84);
    v14 = sub_1F58D20((__int64)v84);
    v15 = v76;
    if ( v14 )
      v7 = sub_1F596B0((__int64)v84);
  }
  v16 = *(_QWORD *)(v5 + 40);
  v87 = v15;
  LOBYTE(v86) = v7;
  v17 = *(_BYTE *)v16;
  v18 = *(_QWORD *)(v16 + 8);
  LOBYTE(v96) = v17;
  v63 = v18;
  v97 = v18;
  if ( v17 )
  {
    if ( (unsigned __int8)(v17 - 14) <= 0x5Fu )
    {
      switch ( v17 )
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
          v17 = 3;
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
          v17 = 4;
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
          v17 = 5;
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
          v17 = 6;
          break;
        case 0x37u:
          v17 = 7;
          break;
        case 0x56u:
        case 0x57u:
        case 0x58u:
        case 0x62u:
        case 0x63u:
        case 0x64u:
          v17 = 8;
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
          v17 = 9;
          break;
        case 0x5Eu:
        case 0x5Fu:
        case 0x60u:
        case 0x61u:
        case 0x6Au:
        case 0x6Bu:
        case 0x6Cu:
        case 0x6Du:
          v17 = 10;
          break;
        default:
          v17 = 2;
          break;
      }
      v63 = 0;
    }
  }
  else
  {
    v55 = sub_1F58D20((__int64)&v96);
    v17 = 0;
    if ( v55 )
    {
      v56 = sub_1F596B0((__int64)&v96);
      v7 = v86;
      v63 = v57;
      v17 = v56;
    }
  }
  v61 = v17;
  if ( v7 )
    v19 = sub_1F3E310(&v86);
  else
    v19 = sub_1F58D40((__int64)&v86);
  v20 = v80;
  v64 = v19 >> 3;
  v72 = v78;
  v21 = *(unsigned __int8 *)(v80[5] + 16LL * v78);
  v22 = *(const void ***)(v80[5] + 16LL * v78 + 8);
  v93 = v95;
  v94 = 0x800000000LL;
  v97 = 0x800000000LL;
  v77 = v21;
  v96 = v98;
  if ( v65 )
  {
    v75 = v22;
    v23 = v11.m128i_u64[1];
    v60 = v19 >> 3;
    v79 = v5;
    v24 = 0;
    v81 = 0;
    while ( 1 )
    {
      v69 = (__int64)v20;
      v37 = *(_QWORD *)(v79 + 104);
      a1 = _mm_loadu_si128((const __m128i *)(v37 + 40));
      v90 = a1;
      v91 = *(__int64 **)(v37 + 56);
      v66 = *(_WORD *)(v37 + 32);
      v38 = sub_1E34390(v37);
      v39 = *(_QWORD *)(v79 + 104);
      v40 = -(__int64)(v24 | v38) & (v24 | v38);
      v41 = *(_QWORD *)v39 & 0xFFFFFFFFFFFFFFF8LL;
      if ( v41 )
      {
        v42 = *(_QWORD *)(v39 + 8) + v24;
        v43 = *(_BYTE *)(v39 + 16);
        if ( (*(_QWORD *)v39 & 4) != 0 )
        {
          *((_QWORD *)&v88 + 1) = *(_QWORD *)(v39 + 8) + v24;
          LOBYTE(v89) = v43;
          *(_QWORD *)&v88 = v41 | 4;
          HIDWORD(v89) = *(_DWORD *)(v41 + 12);
        }
        else
        {
          v44 = *(_QWORD *)v41;
          *(_QWORD *)&v88 = *(_QWORD *)v39 & 0xFFFFFFFFFFFFFFF8LL;
          *((_QWORD *)&v88 + 1) = v42;
          v45 = *(_BYTE *)(v44 + 8) == 16;
          LOBYTE(v89) = v43;
          if ( v45 )
            v44 = **(_QWORD **)(v44 + 16);
          HIDWORD(v89) = *(_DWORD *)(v44 + 8) >> 8;
        }
      }
      else
      {
        v89 = 0;
        v25 = *(_DWORD *)(v39 + 20);
        v88 = 0u;
        HIDWORD(v89) = v25;
      }
      v26 = v23 & 0xFFFFFFFF00000000LL | v72;
      v73 = sub_1D2B810(a4, v62, (__int64)&v82, v61, v63, v40, v10, v69, v26, v88, v89, v86, v87, v66, (__int64)&v90);
      *(_QWORD *)&v27 = sub_1D38BB0((__int64)a4, v60, (__int64)&v82, v77, v75, 0, a1, *(double *)&v10, v11, 0);
      v28 = sub_1D332F0(
              a4,
              52,
              (__int64)&v82,
              v77,
              v75,
              0,
              *(double *)a1.m128i_i64,
              *(double *)&v10,
              v11,
              v69,
              v26,
              v27);
      v30 = v73;
      v20 = v28;
      v31 = v29;
      v23 = v29 | v26 & 0xFFFFFFFF00000000LL;
      v32 = (unsigned int)v94;
      if ( (unsigned int)v94 >= HIDWORD(v94) )
      {
        v71 = v29;
        v68 = v20;
        sub_16CD150((__int64)&v93, v95, 0, 16, (int)v20, v29);
        v32 = (unsigned int)v94;
        v31 = v71;
        v20 = v68;
        v30 = v73;
      }
      v33 = (__int64 *)&v93[16 * v32];
      *v33 = v30;
      v33[1] = 0;
      v34 = (unsigned int)v97;
      LODWORD(v94) = v94 + 1;
      if ( (unsigned int)v97 >= HIDWORD(v97) )
      {
        v70 = v31;
        v67 = v20;
        v74 = v30;
        sub_16CD150((__int64)&v96, v98, 0, 16, (int)v20, v31);
        v34 = (unsigned int)v97;
        v31 = v70;
        v20 = v67;
        v30 = v74;
      }
      v35 = (__int64 *)&v96[16 * v34];
      ++v81;
      v24 += v64;
      *v35 = v30;
      v35[1] = 1;
      v36 = (unsigned int)(v97 + 1);
      LODWORD(v97) = v97 + 1;
      if ( v81 == v65 )
        break;
      v72 = v31;
    }
    v5 = v79;
    v46 = v96;
  }
  else
  {
    v46 = v98;
    v36 = 0;
  }
  *((_QWORD *)&v59 + 1) = v36;
  *(_QWORD *)&v59 = v46;
  v47 = sub_1D359D0(a4, 2, (__int64)&v82, 1, 0, 0, *(double *)a1.m128i_i64, *(double *)&v10, v11, v59);
  v49 = v48;
  *((_QWORD *)&v58 + 1) = (unsigned int)v94;
  *(_QWORD *)&v58 = v93;
  v50 = sub_1D359D0(
          a4,
          104,
          (__int64)&v82,
          **(unsigned __int8 **)(v5 + 40),
          *(const void ***)(*(_QWORD *)(v5 + 40) + 8LL),
          0,
          *(double *)a1.m128i_i64,
          *(double *)&v10,
          v11,
          v58);
  v90.m128i_i32[2] = v51;
  v91 = v47;
  v92 = v49;
  v90.m128i_i64[0] = (__int64)v50;
  v53 = sub_1D37190((__int64)a4, (__int64)&v90, 2u, (__int64)&v82, v52, *(double *)a1.m128i_i64, *(double *)&v10, v11);
  if ( v96 != v98 )
    _libc_free((unsigned __int64)v96);
  if ( v93 != v95 )
    _libc_free((unsigned __int64)v93);
  if ( v82 )
    sub_161E7C0((__int64)&v82, v82);
  return v53;
}
