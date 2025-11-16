// Function: sub_36EC510
// Address: 0x36ec510
//
void __fastcall sub_36EC510(__int64 a1, __int64 a2)
{
  __int64 v2; // r13
  __int64 v3; // r12
  __int64 v4; // rax
  _DWORD *v5; // rax
  __int64 v6; // rsi
  __int64 v7; // rax
  __int64 v8; // rdi
  unsigned __int64 v9; // rbx
  __int64 v10; // rax
  __int64 v11; // r8
  __int64 v12; // rdi
  unsigned int v13; // eax
  unsigned int v14; // edx
  __int64 v15; // r9
  __int64 v16; // r10
  __m128i v17; // xmm2
  __m128i v18; // xmm1
  __int64 v19; // r9
  __m128i v20; // xmm0
  unsigned __int64 *v21; // rcx
  __int64 v22; // r14
  unsigned __int64 **v23; // rax
  __int64 v24; // rdx
  __m128i v25; // xmm0
  __int64 v26; // r13
  __int64 v27; // rdi
  unsigned __int64 v28; // rcx
  unsigned __int64 v29; // r10
  unsigned __int64 v30; // r11
  __int64 v31; // rax
  __int16 v32; // r14
  __m128i v33; // xmm0
  char v34; // si
  unsigned int v35; // edx
  int v36; // esi
  __int64 v37; // r9
  unsigned int v38; // eax
  unsigned int v39; // ecx
  char v40; // r8
  int v41; // esi
  __m128i v42; // xmm0
  unsigned __int64 v43; // r10
  __int64 v44; // r14
  unsigned __int64 **v45; // r11
  __int64 v46; // r13
  int v47; // r12d
  unsigned __int64 v48; // rbx
  __m128i v49; // xmm0
  __int64 v50; // r14
  __m128i v51; // xmm0
  int v52; // r14d
  __m128i v53; // xmm0
  const __m128i *v54; // rdx
  __int64 v55; // rax
  __m128i v56; // xmm0
  __int64 v57; // r14
  __int64 v58; // rcx
  __int64 v59; // r8
  __int64 v60; // r9
  _BOOL4 v61; // esi
  _BOOL4 v62; // esi
  _BOOL4 v63; // esi
  _BOOL4 v64; // esi
  _BOOL4 v65; // esi
  _BOOL4 v66; // esi
  _BOOL4 v67; // esi
  _BOOL4 v68; // esi
  _BOOL4 v69; // esi
  _BOOL4 v70; // esi
  __int64 v71; // [rsp+8h] [rbp-128h]
  __int64 v72; // [rsp+8h] [rbp-128h]
  __m128i v73; // [rsp+10h] [rbp-120h] BYREF
  unsigned __int64 **v74; // [rsp+20h] [rbp-110h]
  __int64 v75; // [rsp+28h] [rbp-108h]
  __m128i v76; // [rsp+30h] [rbp-100h] BYREF
  __m128i v77; // [rsp+40h] [rbp-F0h] BYREF
  __m128i v78; // [rsp+50h] [rbp-E0h] BYREF
  unsigned __int64 v79; // [rsp+60h] [rbp-D0h]
  int v80; // [rsp+68h] [rbp-C8h]
  char v81; // [rsp+6Eh] [rbp-C2h]
  char v82; // [rsp+6Fh] [rbp-C1h]
  __int64 v83; // [rsp+70h] [rbp-C0h] BYREF
  int v84; // [rsp+78h] [rbp-B8h]
  __m128i v85; // [rsp+80h] [rbp-B0h]
  __m128i v86; // [rsp+90h] [rbp-A0h]
  __m128i v87; // [rsp+A0h] [rbp-90h]
  unsigned __int64 *v88; // [rsp+B0h] [rbp-80h] BYREF
  __int64 v89; // [rsp+B8h] [rbp-78h]
  _OWORD v90[7]; // [rsp+C0h] [rbp-70h] BYREF

  v2 = a1;
  v3 = a2;
  v4 = sub_2E79000(*(__int64 **)(*(_QWORD *)(a1 + 64) + 40LL));
  v5 = sub_AE2980(v4, 3u);
  v6 = *(_QWORD *)(a2 + 40);
  v80 = v5[1];
  v7 = *(_QWORD *)(*(_QWORD *)(v6 + 40) + 96LL);
  if ( *(_DWORD *)(v7 + 32) <= 0x40u )
    v8 = *(_QWORD *)(v7 + 24);
  else
    v8 = **(_QWORD **)(v7 + 24);
  v9 = sub_36D7030(v8);
  v10 = *(_QWORD *)(*(_QWORD *)(v6 + 80) + 96LL);
  if ( *(_DWORD *)(v10 + 32) <= 0x40u )
    v11 = *(_QWORD *)(v10 + 24);
  else
    v11 = **(_QWORD **)(v10 + 24);
  v81 = v11;
  v78.m128i_i8[0] = v11 & 1;
  v77.m128i_i8[0] = v11 & 2;
  v82 = v11 & 0x1C;
  LOBYTE(v75) = v11 & 0x18;
  if ( (v11 & 0x18) != 8 )
  {
    if ( v82 != 4 || v9 > 2 )
      goto LABEL_15;
LABEL_451:
    sub_C64ED0("NumDims should be at least 3 for Im2Col or Im2Col_W or Im2Col_W128 mode", 1u);
  }
  v12 = *(_QWORD *)(v2 + 1136);
  v13 = *(_DWORD *)(v12 + 340);
  if ( v13 > 0x408 )
  {
    if ( v13 - 1101 > 1 )
      goto LABEL_8;
  }
  else if ( v13 <= 0x3E8 || ((1LL << ((unsigned __int8)v13 + 23)) & 0xC0000C03) == 0 )
  {
    goto LABEL_8;
  }
  v14 = *(_DWORD *)(v12 + 336);
  if ( (__ROR4__(-858993459 * v13 + 1717986918, 1) > 0x19999999u || v14 <= 0x57) && v14 <= 0x55 )
LABEL_8:
    sub_C64ED0("Im2Col_W and Im2Col_W128 modes are not supported on this architecture.", 1u);
  if ( ((v11 & 0x14) == 4 || v82 == 8) && v9 <= 2 )
    goto LABEL_451;
LABEL_15:
  v15 = *(_QWORD *)(v3 + 80);
  v83 = v15;
  if ( v15 )
  {
    v79 = v11;
    sub_B96E90((__int64)&v83, v15, 1);
    v6 = *(_QWORD *)(v3 + 40);
    v11 = v79;
  }
  v16 = 2;
  v17 = _mm_loadu_si128((const __m128i *)(v6 + 120));
  v18 = _mm_loadu_si128((const __m128i *)(v6 + 160));
  if ( v9 >= 2 )
    v16 = v9;
  v19 = v9 + 6;
  v20 = _mm_loadu_si128((const __m128i *)(v6 + 200));
  v84 = *(_DWORD *)(v3 + 72);
  v88 = (unsigned __int64 *)v90;
  v89 = 0x400000003LL;
  v79 = v16;
  v85 = v17;
  v86 = v18;
  v87 = v20;
  v90[0] = v17;
  v90[1] = v18;
  v90[2] = v20;
  if ( v9 )
  {
    v21 = (unsigned __int64 *)v90;
    v22 = 0;
    v23 = &v88;
    v24 = 3;
    v76.m128i_i64[0] = v2;
    v25 = _mm_loadu_si128((const __m128i *)(v6 + 240));
    v26 = v11;
    while ( 1 )
    {
      ++v22;
      *(__m128i *)&v21[2 * v24] = v25;
      v24 = (unsigned int)(v89 + 1);
      LODWORD(v89) = v89 + 1;
      if ( v9 == v22 )
        break;
      v25 = _mm_loadu_si128((const __m128i *)(*(_QWORD *)(v3 + 40) + 40LL * (unsigned int)(v22 + 6)));
      if ( v24 + 1 > (unsigned __int64)HIDWORD(v89) )
      {
        v71 = v19;
        v74 = v23;
        v73 = v25;
        sub_C8D5F0((__int64)v23, v90, v24 + 1, 0x10u, v11, v19);
        v24 = (unsigned int)v89;
        v19 = v71;
        v25 = _mm_load_si128(&v73);
        v23 = v74;
      }
      v21 = v88;
    }
    v27 = *(_QWORD *)(v3 + 40);
    v28 = HIDWORD(v89);
    v11 = v26;
    v29 = v24 + 1;
    v2 = v76.m128i_i64[0];
    v6 = v27;
    v30 = HIDWORD(v89);
    v76.m128i_i64[0] = v79 - 2 + v19;
    v31 = v24;
    if ( v82 == 4 )
    {
      if ( v79 != 2 )
      {
        v43 = v79;
        v19 = (unsigned int)(v19 - 1);
        v44 = 1;
        v79 = v9;
        v75 = v2;
        v45 = &v88;
        v46 = v3;
        v47 = v19;
        v48 = v43 - 1;
        while ( 1 )
        {
          v49 = _mm_loadu_si128((const __m128i *)(v27 + 40LL * (unsigned int)(v47 + v44)));
          if ( v24 + 1 > v28 )
          {
            v72 = v11;
            v74 = v45;
            v73 = v49;
            sub_C8D5F0((__int64)v45, v90, v24 + 1, 0x10u, v11, v19);
            v24 = (unsigned int)v89;
            v11 = v72;
            v49 = _mm_load_si128(&v73);
            v45 = v74;
          }
          ++v44;
          *(__m128i *)&v88[2 * v24] = v49;
          v24 = (unsigned int)(v89 + 1);
          LODWORD(v89) = v89 + 1;
          if ( v48 == v44 )
            break;
          v27 = *(_QWORD *)(v46 + 40);
          v28 = HIDWORD(v89);
        }
        v3 = v46;
        v31 = (unsigned int)v24;
        v30 = HIDWORD(v89);
        v9 = v79;
        v2 = v75;
        v29 = (unsigned int)v24 + 1LL;
        v6 = *(_QWORD *)(v3 + 40);
        if ( !v77.m128i_i8[0] )
          goto LABEL_28;
LABEL_89:
        v50 = 40LL * v76.m128i_u32[0];
        v51 = _mm_loadu_si128((const __m128i *)(v6 + v50));
        if ( v30 < v29 )
        {
          v79 = v11;
          v77 = v51;
          sub_C8D5F0((__int64)&v88, v90, v29, 0x10u, v11, v19);
          v31 = (unsigned int)v89;
          v51 = _mm_load_si128(&v77);
          v11 = v79;
        }
        goto LABEL_91;
      }
LABEL_27:
      if ( !v77.m128i_i8[0] )
      {
LABEL_28:
        v32 = 0;
LABEL_29:
        if ( !v78.m128i_i8[0] )
          goto LABEL_30;
        v42 = _mm_loadu_si128((const __m128i *)(v6 + 40LL * (unsigned int)(v76.m128i_i32[0] + 1)));
        if ( v29 > v30 )
        {
          v79 = v11;
          v78 = v42;
          sub_C8D5F0((__int64)&v88, v90, v29, 0x10u, v11, v19);
          v31 = (unsigned int)v89;
          v42 = _mm_load_si128(&v78);
          v11 = v79;
        }
        goto LABEL_81;
      }
      goto LABEL_89;
    }
    if ( (_BYTE)v75 != 8 )
      goto LABEL_27;
    v52 = v19;
    v19 = (unsigned int)v19;
    v53 = _mm_loadu_si128((const __m128i *)(v27 + 40LL * (unsigned int)v19));
    if ( v29 > HIDWORD(v89) )
    {
      v79 = v11;
      v76 = v53;
      sub_C8D5F0((__int64)&v88, v90, v29, 0x10u, v11, (unsigned int)v19);
      v31 = (unsigned int)v89;
      v53 = _mm_load_si128(&v76);
      v11 = v79;
    }
LABEL_103:
    *(__m128i *)&v88[2 * v31] = v53;
    v54 = (const __m128i *)(*(_QWORD *)(v3 + 40) + 40LL * (unsigned int)(v52 + 1));
    LODWORD(v89) = v89 + 1;
    v55 = (unsigned int)v89;
    v56 = _mm_loadu_si128(v54);
    if ( (unsigned __int64)(unsigned int)v89 + 1 > HIDWORD(v89) )
    {
      v79 = v11;
      v76 = v56;
      sub_C8D5F0((__int64)&v88, v90, (unsigned int)v89 + 1LL, 0x10u, v11, v19);
      v55 = (unsigned int)v89;
      v56 = _mm_load_si128(&v76);
      v11 = v79;
    }
    *(__m128i *)&v88[2 * v55] = v56;
    v6 = *(_QWORD *)(v3 + 40);
    v76.m128i_i64[0] = v9 + 8;
    v30 = HIDWORD(v89);
    LODWORD(v89) = v89 + 1;
    v31 = (unsigned int)v89;
    v29 = (unsigned int)v89 + 1LL;
    goto LABEL_27;
  }
  if ( v82 != 4 )
  {
    if ( (_BYTE)v75 == 8 )
    {
      v53 = _mm_loadu_si128((const __m128i *)(v6 + 240));
      v52 = 6;
      v31 = 3;
      goto LABEL_103;
    }
    if ( !v77.m128i_i8[0] )
    {
      v76.m128i_i64[0] = 6;
      v29 = 4;
      v31 = 3;
      v30 = 4;
      goto LABEL_28;
    }
    v76.m128i_i64[0] = 6;
    v51 = _mm_loadu_si128((const __m128i *)(v6 + 240));
    v31 = 3;
    v50 = 240;
LABEL_91:
    *(__m128i *)&v88[2 * v31] = v51;
    v6 = *(_QWORD *)(v3 + 40);
    v30 = HIDWORD(v89);
    LODWORD(v89) = v89 + 1;
    v31 = (unsigned int)v89;
    v29 = (unsigned int)v89 + 1LL;
    v32 = *(_WORD *)(*(_QWORD *)(*(_QWORD *)(v6 + v50) + 48LL) + 16LL * *(unsigned int *)(v6 + v50 + 8));
    goto LABEL_29;
  }
  if ( v77.m128i_i8[0] )
  {
    v76.m128i_i64[0] = 6;
    v50 = 240;
    v31 = 3;
    v51 = _mm_loadu_si128((const __m128i *)(v6 + 240));
    goto LABEL_91;
  }
  if ( v78.m128i_i8[0] )
  {
    v42 = _mm_loadu_si128((const __m128i *)(v6 + 280));
    v32 = 0;
    v31 = 3;
LABEL_81:
    *(__m128i *)&v88[2 * v31] = v42;
    v6 = *(_QWORD *)(v3 + 40);
    v30 = HIDWORD(v89);
    LODWORD(v89) = v89 + 1;
    v31 = (unsigned int)v89;
    v29 = (unsigned int)v89 + 1LL;
    goto LABEL_30;
  }
  v32 = 0;
  v30 = 4;
  v29 = 4;
  v31 = 3;
LABEL_30:
  v33 = _mm_loadu_si128((const __m128i *)v6);
  if ( v30 < v29 )
  {
    v79 = v11;
    v78 = v33;
    sub_C8D5F0((__int64)&v88, v90, v29, 0x10u, v11, v19);
    v31 = (unsigned int)v89;
    v33 = _mm_load_si128(&v78);
    LOBYTE(v11) = v79;
  }
  v34 = v81;
  *(__m128i *)&v88[2 * v31] = v33;
  v35 = v89 + 1;
  LODWORD(v89) = v89 + 1;
  v36 = v34 & 0x20;
  if ( v36 )
  {
    v37 = *(_QWORD *)(v2 + 1136);
    v38 = *(_DWORD *)(v37 + 340);
    if ( v38 > 0x408 )
    {
      if ( v38 - 1101 > 1 )
        goto LABEL_36;
    }
    else if ( v38 <= 0x3E8 || ((1LL << ((unsigned __int8)v38 + 23)) & 0xC0000C03) == 0 )
    {
      goto LABEL_36;
    }
    v39 = *(_DWORD *)(v37 + 336);
    if ( (__ROR4__(-858993459 * v38 + 1717986918, 1) > 0x19999999u || v39 <= 0x57) && v39 <= 0x55 )
LABEL_36:
      sub_C64ED0("2CTA Mode for CpAsyncBulkTensorG2S not supported on this architecture", 1u);
  }
  v40 = v11 & 1;
  switch ( v82 )
  {
    case 4:
      if ( v9 != 4 )
      {
        if ( v9 == 5 )
        {
          if ( (_BYTE)v36 )
          {
            if ( v32 )
            {
              if ( v40 )
              {
                v41 = 851;
                if ( v80 != 32 )
                  v41 = 827;
              }
              else
              {
                v41 = 850;
                if ( v80 != 32 )
                  v41 = 826;
              }
            }
            else if ( v40 )
            {
              v41 = 849;
              if ( v80 != 32 )
                v41 = 825;
            }
            else
            {
              v41 = 848;
              if ( v80 != 32 )
                v41 = 824;
            }
          }
          else if ( v32 )
          {
            if ( v40 )
            {
              v41 = 854;
              if ( v80 != 32 )
                v41 = 830;
            }
            else
            {
              v41 = 853;
              if ( v80 != 32 )
                v41 = 829;
            }
          }
          else if ( v40 )
          {
            v41 = 852;
            if ( v80 != 32 )
              v41 = 828;
          }
          else
          {
            v41 = 847;
            if ( v80 != 32 )
              v41 = 823;
          }
          break;
        }
        if ( v9 == 3 )
        {
          if ( (_BYTE)v36 )
          {
            if ( v32 )
            {
              if ( v40 )
              {
                v41 = 723;
                if ( v80 != 32 )
                  v41 = 699;
              }
              else
              {
                v41 = 722;
                if ( v80 != 32 )
                  v41 = 698;
              }
            }
            else if ( v40 )
            {
              v41 = 721;
              if ( v80 != 32 )
                v41 = 697;
            }
            else
            {
              v41 = 720;
              if ( v80 != 32 )
                v41 = 696;
            }
          }
          else if ( v32 )
          {
            if ( v40 )
            {
              v41 = 726;
              if ( v80 != 32 )
                v41 = 702;
            }
            else
            {
              v41 = 725;
              if ( v80 != 32 )
                v41 = 701;
            }
          }
          else if ( v40 )
          {
            v41 = 724;
            if ( v80 != 32 )
              v41 = 700;
          }
          else
          {
            v41 = 719;
            if ( v80 != 32 )
              v41 = 695;
          }
          break;
        }
LABEL_452:
        BUG();
      }
      if ( (_BYTE)v36 )
      {
        if ( v32 )
        {
          if ( v40 )
          {
            v41 = 787;
            if ( v80 != 32 )
              v41 = 763;
          }
          else
          {
            v41 = 786;
            if ( v80 != 32 )
              v41 = 762;
          }
        }
        else if ( v40 )
        {
          v41 = 785;
          if ( v80 != 32 )
            v41 = 761;
        }
        else
        {
          v41 = 784;
          if ( v80 != 32 )
            v41 = 760;
        }
      }
      else if ( v32 )
      {
        if ( v40 )
        {
          v41 = 790;
          if ( v80 != 32 )
            v41 = 766;
        }
        else
        {
          v41 = 789;
          if ( v80 != 32 )
            v41 = 765;
        }
      }
      else if ( v40 )
      {
        v41 = 788;
        if ( v80 != 32 )
          v41 = 764;
      }
      else
      {
        v41 = 783;
        if ( v80 != 32 )
          v41 = 759;
      }
      break;
    case 8:
      switch ( v9 )
      {
        case 4uLL:
          if ( (_BYTE)v36 )
          {
            if ( v32 )
            {
              if ( v40 )
              {
                v41 = 803;
                if ( v80 != 32 )
                  v41 = 779;
              }
              else
              {
                v41 = 802;
                if ( v80 != 32 )
                  v41 = 778;
              }
            }
            else if ( v40 )
            {
              v41 = 801;
              if ( v80 != 32 )
                v41 = 777;
            }
            else
            {
              v41 = 800;
              if ( v80 != 32 )
                v41 = 776;
            }
          }
          else if ( v32 )
          {
            if ( v40 )
            {
              v41 = 806;
              if ( v80 != 32 )
                v41 = 782;
            }
            else
            {
              v41 = 805;
              if ( v80 != 32 )
                v41 = 781;
            }
          }
          else if ( v40 )
          {
            v41 = 804;
            if ( v80 != 32 )
              v41 = 780;
          }
          else
          {
            v41 = 791;
            if ( v80 != 32 )
              v41 = 767;
          }
          break;
        case 5uLL:
          if ( (_BYTE)v36 )
          {
            if ( v32 )
            {
              if ( v40 )
              {
                v41 = 867;
                if ( v80 != 32 )
                  v41 = 843;
              }
              else
              {
                v41 = 866;
                if ( v80 != 32 )
                  v41 = 842;
              }
            }
            else if ( v40 )
            {
              v41 = 865;
              if ( v80 != 32 )
                v41 = 841;
            }
            else
            {
              v41 = 864;
              if ( v80 != 32 )
                v41 = 840;
            }
          }
          else if ( v32 )
          {
            if ( v40 )
            {
              v41 = 870;
              if ( v80 != 32 )
                v41 = 846;
            }
            else
            {
              v41 = 869;
              if ( v80 != 32 )
                v41 = 845;
            }
          }
          else if ( v40 )
          {
            v41 = 868;
            if ( v80 != 32 )
              v41 = 844;
          }
          else
          {
            v41 = 855;
            if ( v80 != 32 )
              v41 = 831;
          }
          break;
        case 3uLL:
          if ( (_BYTE)v36 )
          {
            if ( v32 )
            {
              if ( v40 )
              {
                v41 = 739;
                if ( v80 != 32 )
                  v41 = 715;
              }
              else
              {
                v41 = 738;
                if ( v80 != 32 )
                  v41 = 714;
              }
            }
            else if ( v40 )
            {
              v41 = 737;
              if ( v80 != 32 )
                v41 = 713;
            }
            else
            {
              v41 = 736;
              if ( v80 != 32 )
                v41 = 712;
            }
          }
          else if ( v32 )
          {
            if ( v40 )
            {
              v41 = 742;
              if ( v80 != 32 )
                v41 = 718;
            }
            else
            {
              v41 = 741;
              if ( v80 != 32 )
                v41 = 717;
            }
          }
          else if ( v40 )
          {
            v41 = 740;
            if ( v80 != 32 )
              v41 = 716;
          }
          else
          {
            v41 = 727;
            if ( v80 != 32 )
              v41 = 703;
          }
          break;
        default:
          goto LABEL_452;
      }
      break;
    case 12:
      switch ( v9 )
      {
        case 4uLL:
          if ( (_BYTE)v36 )
          {
            if ( v32 )
            {
              if ( v40 )
              {
                v41 = 796;
                if ( v80 != 32 )
                  v41 = 772;
              }
              else
              {
                v41 = 795;
                if ( v80 != 32 )
                  v41 = 771;
              }
            }
            else if ( v40 )
            {
              v41 = 794;
              if ( v80 != 32 )
                v41 = 770;
            }
            else
            {
              v41 = 793;
              if ( v80 != 32 )
                v41 = 769;
            }
          }
          else if ( v32 )
          {
            if ( v40 )
            {
              v41 = 799;
              if ( v80 != 32 )
                v41 = 775;
            }
            else
            {
              v41 = 798;
              if ( v80 != 32 )
                v41 = 774;
            }
          }
          else if ( v40 )
          {
            v41 = 797;
            if ( v80 != 32 )
              v41 = 773;
          }
          else
          {
            v41 = 792;
            if ( v80 != 32 )
              v41 = 768;
          }
          break;
        case 5uLL:
          if ( (_BYTE)v36 )
          {
            if ( v32 )
            {
              if ( v40 )
              {
                v41 = 860;
                if ( v80 != 32 )
                  v41 = 836;
              }
              else
              {
                v41 = 859;
                if ( v80 != 32 )
                  v41 = 835;
              }
            }
            else if ( v40 )
            {
              v41 = 858;
              if ( v80 != 32 )
                v41 = 834;
            }
            else
            {
              v41 = 857;
              if ( v80 != 32 )
                v41 = 833;
            }
          }
          else if ( v32 )
          {
            if ( v40 )
            {
              v41 = 863;
              if ( v80 != 32 )
                v41 = 839;
            }
            else
            {
              v41 = 862;
              if ( v80 != 32 )
                v41 = 838;
            }
          }
          else if ( v40 )
          {
            v41 = 861;
            if ( v80 != 32 )
              v41 = 837;
          }
          else
          {
            v41 = 856;
            if ( v80 != 32 )
              v41 = 832;
          }
          break;
        case 3uLL:
          if ( (_BYTE)v36 )
          {
            if ( v32 )
            {
              if ( v40 )
              {
                v41 = 732;
                if ( v80 != 32 )
                  v41 = 708;
              }
              else
              {
                v41 = 731;
                if ( v80 != 32 )
                  v41 = 707;
              }
            }
            else if ( v40 )
            {
              v41 = 730;
              if ( v80 != 32 )
                v41 = 706;
            }
            else
            {
              v41 = 729;
              if ( v80 != 32 )
                v41 = 705;
            }
          }
          else if ( v32 )
          {
            if ( v40 )
            {
              v41 = 735;
              if ( v80 != 32 )
                v41 = 711;
            }
            else
            {
              v41 = 734;
              if ( v80 != 32 )
                v41 = 710;
            }
          }
          else if ( v40 )
          {
            v41 = 733;
            if ( v80 != 32 )
              v41 = 709;
          }
          else
          {
            v41 = 728;
            if ( v80 != 32 )
              v41 = 704;
          }
          break;
        default:
          goto LABEL_452;
      }
      break;
    default:
      switch ( v9 )
      {
        case 1uLL:
          if ( (_BYTE)v36 )
          {
            if ( v32 )
            {
              v64 = v80 != 32;
              if ( v40 )
                v41 = 8 * v64 + 667;
              else
                v41 = 8 * v64 + 666;
            }
            else if ( v40 )
            {
              v41 = 8 * (v80 != 32) + 665;
            }
            else
            {
              v41 = 8 * (v80 != 32) + 664;
            }
          }
          else if ( v32 )
          {
            v70 = v80 != 32;
            if ( v40 )
              v41 = 8 * v70 + 670;
            else
              v41 = 8 * v70 + 669;
          }
          else if ( v40 )
          {
            v41 = 8 * (v80 != 32) + 668;
          }
          else
          {
            v41 = 8 * (v80 != 32) + 663;
          }
          goto LABEL_148;
        case 2uLL:
          if ( (_BYTE)v36 )
          {
            if ( v32 )
            {
              v62 = v80 != 32;
              if ( v40 )
                v41 = 8 * v62 + 683;
              else
                v41 = 8 * v62 + 682;
            }
            else if ( v40 )
            {
              v41 = 8 * (v80 != 32) + 681;
            }
            else
            {
              v41 = 8 * (v80 != 32) + 680;
            }
          }
          else if ( v32 )
          {
            v67 = v80 != 32;
            if ( v40 )
              v41 = 8 * v67 + 686;
            else
              v41 = 8 * v67 + 685;
          }
          else if ( v40 )
          {
            v41 = 8 * (v80 != 32) + 684;
          }
          else
          {
            v41 = 8 * (v80 != 32) + 679;
          }
          goto LABEL_148;
        case 3uLL:
          if ( (_BYTE)v36 )
          {
            if ( v32 )
            {
              v65 = v80 != 32;
              if ( v40 )
                v41 = 8 * v65 + 747;
              else
                v41 = 8 * v65 + 746;
            }
            else if ( v40 )
            {
              v41 = 8 * (v80 != 32) + 745;
            }
            else
            {
              v41 = 8 * (v80 != 32) + 744;
            }
          }
          else if ( v32 )
          {
            v66 = v80 != 32;
            if ( v40 )
              v41 = 8 * v66 + 750;
            else
              v41 = 8 * v66 + 749;
          }
          else if ( v40 )
          {
            v41 = 8 * (v80 != 32) + 748;
          }
          else
          {
            v41 = 8 * (v80 != 32) + 743;
          }
          goto LABEL_148;
        case 4uLL:
          if ( (_BYTE)v36 )
          {
            if ( v32 )
            {
              v63 = v80 != 32;
              if ( v40 )
                v41 = 8 * v63 + 811;
              else
                v41 = 8 * v63 + 810;
            }
            else if ( v40 )
            {
              v41 = 8 * (v80 != 32) + 809;
            }
            else
            {
              v41 = 8 * (v80 != 32) + 808;
            }
          }
          else if ( v32 )
          {
            v69 = v80 != 32;
            if ( v40 )
              v41 = 8 * v69 + 814;
            else
              v41 = 8 * v69 + 813;
          }
          else if ( v40 )
          {
            v41 = 8 * (v80 != 32) + 812;
          }
          else
          {
            v41 = 8 * (v80 != 32) + 807;
          }
          goto LABEL_148;
        case 5uLL:
          if ( (_BYTE)v36 )
          {
            if ( v32 )
            {
              v61 = v80 != 32;
              if ( v40 )
                v41 = 8 * v61 + 875;
              else
                v41 = 8 * v61 + 874;
            }
            else if ( v40 )
            {
              v41 = 8 * (v80 != 32) + 873;
            }
            else
            {
              v41 = 8 * (v80 != 32) + 872;
            }
          }
          else if ( v32 )
          {
            v68 = v80 != 32;
            if ( v40 )
              v41 = 8 * v68 + 878;
            else
              v41 = 8 * v68 + 877;
          }
          else if ( v40 )
          {
            v41 = 8 * (v80 != 32) + 876;
          }
          else
          {
            v41 = 8 * (v80 != 32) + 871;
          }
          goto LABEL_148;
        default:
          goto LABEL_452;
      }
  }
LABEL_148:
  v57 = sub_33E66D0(
          *(_QWORD **)(v2 + 64),
          v41,
          (__int64)&v83,
          *(_QWORD *)(v3 + 48),
          *(unsigned int *)(v3 + 68),
          v35,
          v88,
          v35);
  sub_34158F0(*(_QWORD *)(v2 + 64), v3, v57, v58, v59, v60);
  sub_3421DB0(v57);
  sub_33ECEA0(*(const __m128i **)(v2 + 64), v3);
  if ( v88 != (unsigned __int64 *)v90 )
    _libc_free((unsigned __int64)v88);
  if ( v83 )
    sub_B91220((__int64)&v83, v83);
}
