// Function: sub_22A87A0
// Address: 0x22a87a0
//
char __fastcall sub_22A87A0(__int64 a1, __int64 a2, __int64 a3)
{
  __int64 v4; // r13
  __int64 v6; // rdx
  __int64 v7; // r12
  unsigned __int8 v8; // al
  bool v9; // cc
  char *v10; // rsi
  size_t v11; // rdx
  _DWORD *v12; // rax
  _BYTE *v13; // rax
  _QWORD *v14; // rdx
  __int64 v15; // rax
  char *v16; // rax
  size_t v17; // rdx
  _BYTE *v18; // rdi
  unsigned __int8 *v19; // rsi
  _BYTE *v20; // rax
  size_t v21; // r15
  __m128i *v22; // rdx
  unsigned __int64 v23; // rsi
  __int64 v24; // rdi
  _BYTE *v25; // rax
  __m128i *v26; // rdx
  int v27; // eax
  size_t v28; // rdx
  char *v29; // rsi
  _QWORD *v30; // rax
  __m128i *v31; // rdx
  __m128i v32; // xmm0
  int v33; // eax
  size_t v34; // r8
  char *v35; // rsi
  _QWORD *v36; // rax
  __int64 v37; // rax
  __m128i *v38; // rdx
  unsigned __int64 v39; // rbx
  __m128i v40; // xmm0
  __int64 v41; // rdi
  __int64 v42; // rdi
  _BYTE *v43; // rax
  void *v44; // rdx
  __int64 v45; // rax
  __m128i *v46; // rdx
  __m128i *v47; // rdx
  __int64 v48; // r12
  unsigned int v49; // eax
  __int64 v50; // rdi
  _BYTE *v51; // rax
  unsigned __int64 v52; // rax
  __m128i *v53; // rdx
  __m128i si128; // xmm0
  __int64 v55; // rdi
  __int64 v56; // rdi
  _BYTE *v57; // rax
  void *v58; // rdx
  __int64 v59; // rdi
  _BYTE *v60; // rax
  __int64 v61; // rdx
  __int64 v62; // rdi
  _BYTE *v63; // rax
  unsigned __int64 v64; // r8
  char *v65; // rax
  char *v66; // rsi
  unsigned int v67; // ecx
  unsigned int v68; // eax
  __int64 v69; // rdi
  size_t v70; // r8
  char *v71; // rsi
  _QWORD *v72; // rax
  __int64 v73; // rax
  _BYTE *v74; // rax
  __int64 v75; // rdx
  __m128i v76; // xmm0
  __int64 v77; // rax
  __int64 v78; // rax
  __int64 v79; // rax
  unsigned __int64 v80; // rdi
  char *v81; // rax
  char *v82; // rsi
  unsigned int v83; // edx
  unsigned int v84; // eax
  __int64 v85; // rcx
  __int64 v86; // rax
  unsigned __int64 v87; // rdi
  char *v88; // rax
  char *v89; // rsi
  unsigned int v90; // edx
  unsigned int v91; // eax
  __int64 v92; // rcx
  __int64 v93; // rax
  __int16 v95; // [rsp+Dh] [rbp-33h]
  unsigned __int8 v96; // [rsp+Fh] [rbp-31h]

  v4 = a2;
  v6 = *(_QWORD *)(a2 + 32);
  if ( (unsigned __int64)(*(_QWORD *)(a2 + 24) - v6) > 8 )
  {
    *(_BYTE *)(v6 + 8) = 32;
    v7 = a2;
    *(_QWORD *)v6 = 0x3A7373616C432020LL;
    *(_QWORD *)(a2 + 32) += 9LL;
    v8 = *(_BYTE *)(a1 + 10);
    v9 = v8 <= 2u;
    if ( v8 != 2 )
      goto LABEL_3;
LABEL_43:
    v11 = 7;
    v10 = "CBuffer";
    goto LABEL_6;
  }
  v7 = sub_CB6200(a2, "  Class: ", 9u);
  v8 = *(_BYTE *)(a1 + 10);
  v9 = v8 <= 2u;
  if ( v8 == 2 )
    goto LABEL_43;
LABEL_3:
  if ( v9 )
  {
    v10 = (char *)&unk_436634C;
    v11 = 3;
    if ( !v8 )
      v10 = (char *)&unk_4366348;
  }
  else
  {
    if ( v8 != 3 )
      goto LABEL_157;
    v11 = 7;
    v10 = "Sampler";
  }
LABEL_6:
  v12 = *(_DWORD **)(v7 + 32);
  if ( *(_QWORD *)(v7 + 24) - (_QWORD)v12 < v11 )
  {
    v7 = sub_CB6200(v7, (unsigned __int8 *)v10, v11);
    v13 = *(_BYTE **)(v7 + 32);
    if ( *(_BYTE **)(v7 + 24) == v13 )
      goto LABEL_10;
LABEL_15:
    *v13 = 10;
    v14 = (_QWORD *)(*(_QWORD *)(v7 + 32) + 1LL);
    v15 = *(_QWORD *)(v7 + 24);
    *(_QWORD *)(v7 + 32) = v14;
    if ( (unsigned __int64)(v15 - (_QWORD)v14) <= 7 )
      goto LABEL_11;
    goto LABEL_16;
  }
  if ( (v11 & 4) != 0 )
  {
    *v12 = *(_DWORD *)v10;
    *(_DWORD *)((char *)v12 + (unsigned int)v11 - 4) = *(_DWORD *)&v10[(unsigned int)v11 - 4];
  }
  else
  {
    *(_BYTE *)v12 = *v10;
    *(_WORD *)((char *)v12 + (unsigned int)v11 - 2) = *(_WORD *)&v10[(unsigned int)v11 - 2];
  }
  v13 = (_BYTE *)(v11 + *(_QWORD *)(v7 + 32));
  *(_QWORD *)(v7 + 32) = v13;
  if ( *(_BYTE **)(v7 + 24) != v13 )
    goto LABEL_15;
LABEL_10:
  v7 = sub_CB6200(v7, (unsigned __int8 *)"\n", 1u);
  v14 = *(_QWORD **)(v7 + 32);
  if ( *(_QWORD *)(v7 + 24) - (_QWORD)v14 <= 7u )
  {
LABEL_11:
    v7 = sub_CB6200(v7, "  Kind: ", 8u);
    goto LABEL_17;
  }
LABEL_16:
  *v14 = 0x203A646E694B2020LL;
  *(_QWORD *)(v7 + 32) += 8LL;
LABEL_17:
  v16 = sub_22A6120(*(_DWORD *)(a1 + 12));
  v18 = *(_BYTE **)(v7 + 32);
  v19 = (unsigned __int8 *)v16;
  v20 = *(_BYTE **)(v7 + 24);
  v21 = v17;
  if ( v20 - v18 < v17 )
  {
    v7 = sub_CB6200(v7, v19, v17);
    v20 = *(_BYTE **)(v7 + 24);
    v18 = *(_BYTE **)(v7 + 32);
  }
  else if ( v17 )
  {
    memcpy(v18, v19, v17);
    v20 = *(_BYTE **)(v7 + 24);
    v18 = (_BYTE *)(v21 + *(_QWORD *)(v7 + 32));
    *(_QWORD *)(v7 + 32) = v18;
  }
  if ( v20 == v18 )
  {
    sub_CB6200(v7, (unsigned __int8 *)"\n", 1u);
  }
  else
  {
    *v18 = 10;
    ++*(_QWORD *)(v7 + 32);
  }
  if ( sub_22A6B90(a1) )
  {
    v22 = *(__m128i **)(v4 + 32);
    if ( *(_QWORD *)(v4 + 24) - (_QWORD)v22 <= 0xFu )
    {
      v4 = sub_CB6200(v4, "  CBuffer size: ", 0x10u);
    }
    else
    {
      *v22 = _mm_load_si128((const __m128i *)&xmmword_4366560);
      *(_QWORD *)(v4 + 32) += 16LL;
    }
    v23 = (unsigned int)sub_22A6C90(a1, a3);
LABEL_26:
    v24 = sub_CB59D0(v4, v23);
    v25 = *(_BYTE **)(v24 + 32);
    if ( *(_BYTE **)(v24 + 24) != v25 )
    {
      *v25 = 10;
      ++*(_QWORD *)(v24 + 32);
      return (char)v25;
    }
    goto LABEL_57;
  }
  if ( !sub_22A6BA0(a1) )
  {
    if ( sub_22A6B80(a1) )
    {
      v52 = sub_22A6C20(a1);
      v95 = v52;
      v53 = *(__m128i **)(v4 + 32);
      v96 = BYTE2(v52);
      if ( *(_QWORD *)(v4 + 24) - (_QWORD)v53 <= 0x14u )
      {
        v55 = sub_CB6200(v4, "  Globally Coherent: ", 0x15u);
      }
      else
      {
        si128 = _mm_load_si128((const __m128i *)&xmmword_4366580);
        v53[1].m128i_i32[0] = 980708965;
        v55 = v4;
        v53[1].m128i_i8[4] = 32;
        *v53 = si128;
        *(_QWORD *)(v4 + 32) += 21LL;
      }
      v56 = sub_CB59F0(v55, (unsigned __int8)v95);
      v57 = *(_BYTE **)(v56 + 32);
      if ( *(_BYTE **)(v56 + 24) == v57 )
      {
        v77 = sub_CB6200(v56, (unsigned __int8 *)"\n", 1u);
        v58 = *(void **)(v77 + 32);
        v56 = v77;
      }
      else
      {
        *v57 = 10;
        v58 = (void *)(*(_QWORD *)(v56 + 32) + 1LL);
        *(_QWORD *)(v56 + 32) = v58;
      }
      if ( *(_QWORD *)(v56 + 24) - (_QWORD)v58 <= 0xDu )
      {
        v56 = sub_CB6200(v56, "  HasCounter: ", 0xEu);
      }
      else
      {
        qmemcpy(v58, "  HasCounter: ", 14);
        *(_QWORD *)(v56 + 32) += 14LL;
      }
      v59 = sub_CB59F0(v56, HIBYTE(v95));
      v60 = *(_BYTE **)(v59 + 32);
      if ( *(_BYTE **)(v59 + 24) == v60 )
      {
        v78 = sub_CB6200(v59, (unsigned __int8 *)"\n", 1u);
        v61 = *(_QWORD *)(v78 + 32);
        v59 = v78;
      }
      else
      {
        *v60 = 10;
        v61 = *(_QWORD *)(v59 + 32) + 1LL;
        *(_QWORD *)(v59 + 32) = v61;
      }
      if ( (unsigned __int64)(*(_QWORD *)(v59 + 24) - v61) <= 8 )
      {
        v59 = sub_CB6200(v59, "  IsROV: ", 9u);
      }
      else
      {
        *(_BYTE *)(v61 + 8) = 32;
        *(_QWORD *)v61 = 0x3A564F5273492020LL;
        *(_QWORD *)(v59 + 32) += 9LL;
      }
      v62 = sub_CB59F0(v59, v96);
      v63 = *(_BYTE **)(v62 + 32);
      if ( *(_BYTE **)(v62 + 24) == v63 )
      {
        sub_CB6200(v62, (unsigned __int8 *)"\n", 1u);
      }
      else
      {
        *v63 = 10;
        ++*(_QWORD *)(v62 + 32);
      }
    }
    if ( (unsigned __int8)sub_22A6C00(a1) )
    {
      v47 = *(__m128i **)(v4 + 32);
      if ( *(_QWORD *)(v4 + 24) - (_QWORD)v47 <= 0xFu )
      {
        v48 = sub_CB6200(v4, "  Sample Count: ", 0x10u);
      }
      else
      {
        v48 = v4;
        *v47 = _mm_load_si128((const __m128i *)&xmmword_4366590);
        *(_QWORD *)(v4 + 32) += 16LL;
      }
      v49 = sub_22A6F10(a1);
      v50 = sub_CB59D0(v48, v49);
      v51 = *(_BYTE **)(v50 + 32);
      if ( *(_BYTE **)(v50 + 24) == v51 )
      {
        sub_CB6200(v50, (unsigned __int8 *)"\n", 1u);
      }
      else
      {
        *v51 = 10;
        ++*(_QWORD *)(v50 + 32);
      }
    }
    if ( sub_22A6BB0(a1) )
    {
      v37 = sub_22A6D40((__int64 *)a1, a3);
      v38 = *(__m128i **)(v4 + 32);
      v39 = v37;
      if ( *(_QWORD *)(v4 + 24) - (_QWORD)v38 <= 0x10u )
      {
        v41 = sub_CB6200(v4, "  Buffer Stride: ", 0x11u);
      }
      else
      {
        v40 = _mm_load_si128((const __m128i *)&xmmword_43665A0);
        v38[1].m128i_i8[0] = 32;
        v41 = v4;
        *v38 = v40;
        *(_QWORD *)(v4 + 32) += 17LL;
      }
      v42 = sub_CB59D0(v41, (unsigned int)v39);
      v43 = *(_BYTE **)(v42 + 32);
      if ( *(_BYTE **)(v42 + 24) == v43 )
      {
        sub_CB6200(v42, (unsigned __int8 *)"\n", 1u);
      }
      else
      {
        *v43 = 10;
        ++*(_QWORD *)(v42 + 32);
      }
      v44 = *(void **)(v4 + 32);
      if ( *(_QWORD *)(v4 + 24) - (_QWORD)v44 <= 0xCu )
      {
        v4 = sub_CB6200(v4, "  Alignment: ", 0xDu);
      }
      else
      {
        qmemcpy(v44, "  Alignment: ", 13);
        *(_QWORD *)(v4 + 32) += 13LL;
      }
LABEL_67:
      v23 = HIDWORD(v39);
      goto LABEL_26;
    }
    if ( !(unsigned __int8)sub_22A6BC0(a1) )
    {
      LOBYTE(v25) = sub_22A6BF0(a1);
      if ( !(_BYTE)v25 )
        return (char)v25;
      v31 = *(__m128i **)(v4 + 32);
      if ( *(_QWORD *)(v4 + 24) - (_QWORD)v31 <= 0x10u )
      {
        v4 = sub_CB6200(v4, "  Feedback Type: ", 0x11u);
      }
      else
      {
        v32 = _mm_load_si128((const __m128i *)&xmmword_43665D0);
        v31[1].m128i_i8[0] = 32;
        *v31 = v32;
        *(_QWORD *)(v4 + 32) += 17LL;
      }
      v33 = sub_22A6F00(a1);
      if ( v33 )
      {
        if ( v33 != 1 )
          goto LABEL_157;
        v34 = 13;
        v35 = "MipRegionUsed";
      }
      else
      {
        v34 = 6;
        v35 = "MinMip";
      }
      v36 = *(_QWORD **)(v4 + 32);
      if ( v34 > *(_QWORD *)(v4 + 24) - (_QWORD)v36 )
      {
        v4 = sub_CB6200(v4, (unsigned __int8 *)v35, v34);
        v25 = *(_BYTE **)(v4 + 32);
      }
      else
      {
        if ( (unsigned int)v34 >= 8 )
        {
          v87 = (unsigned __int64)(v36 + 1) & 0xFFFFFFFFFFFFFFF8LL;
          *v36 = *(_QWORD *)v35;
          *(_QWORD *)((char *)v36 + v34 - 8) = *(_QWORD *)&v35[v34 - 8];
          v88 = (char *)v36 - v87;
          v89 = (char *)(v35 - v88);
          if ( (((_DWORD)v34 + (_DWORD)v88) & 0xFFFFFFF8) >= 8 )
          {
            v90 = (v34 + (_DWORD)v88) & 0xFFFFFFF8;
            v91 = 0;
            do
            {
              v92 = v91;
              v91 += 8;
              *(_QWORD *)(v87 + v92) = *(_QWORD *)&v89[v92];
            }
            while ( v91 < v90 );
          }
          v93 = *(_QWORD *)(v4 + 32);
        }
        else
        {
          *(_DWORD *)v36 = *(_DWORD *)v35;
          *(_DWORD *)((char *)v36 + (unsigned int)v34 - 4) = *(_DWORD *)&v35[(unsigned int)v34 - 4];
          v93 = *(_QWORD *)(v4 + 32);
        }
        v25 = (_BYTE *)(v34 + v93);
        *(_QWORD *)(v4 + 32) = v25;
      }
      if ( v25 != *(_BYTE **)(v4 + 24) )
        goto LABEL_93;
      goto LABEL_56;
    }
    v45 = sub_22A6E00((__int64 *)a1);
    v46 = *(__m128i **)(v4 + 32);
    v39 = v45;
    if ( *(_QWORD *)(v4 + 24) - (_QWORD)v46 <= 0xFu )
    {
      v4 = sub_CB6200(v4, "  Element Type: ", 0x10u);
    }
    else
    {
      *v46 = _mm_load_si128((const __m128i *)&xmmword_43665B0);
      *(_QWORD *)(v4 + 32) += 16LL;
    }
    switch ( (int)v39 )
    {
      case 0:
        v70 = 9;
        v71 = "<invalid>";
        goto LABEL_101;
      case 1:
        v70 = 2;
        v71 = "i1";
        goto LABEL_101;
      case 2:
        v70 = 3;
        v71 = "i16";
        goto LABEL_101;
      case 3:
        v70 = 3;
        v71 = "u16";
        goto LABEL_101;
      case 4:
        v70 = 3;
        v71 = "i32";
        goto LABEL_101;
      case 5:
        v70 = 3;
        v71 = "u32";
        goto LABEL_101;
      case 6:
        v70 = 3;
        v71 = "i64";
        goto LABEL_101;
      case 7:
        v70 = 3;
        v71 = "u64";
        goto LABEL_101;
      case 8:
        v70 = 3;
        v71 = "f16";
        goto LABEL_101;
      case 9:
        v70 = 3;
        v71 = "f32";
        goto LABEL_101;
      case 10:
        v70 = 3;
        v71 = "f64";
        goto LABEL_101;
      case 11:
        v70 = 9;
        v71 = "snorm_f16";
        goto LABEL_101;
      case 12:
        v70 = 9;
        v71 = "unorm_f16";
        goto LABEL_101;
      case 13:
        v70 = 9;
        v71 = "snorm_f32";
        goto LABEL_101;
      case 14:
        v70 = 9;
        v71 = "unorm_f32";
        goto LABEL_101;
      case 15:
        v70 = 9;
        v71 = "snorm_f64";
        goto LABEL_101;
      case 16:
        v70 = 9;
        v71 = "unorm_f64";
        goto LABEL_101;
      case 17:
        v70 = 5;
        v71 = "p32i8";
        goto LABEL_101;
      case 18:
        v70 = 5;
        v71 = "p32u8";
LABEL_101:
        v72 = *(_QWORD **)(v4 + 32);
        if ( *(_QWORD *)(v4 + 24) - (_QWORD)v72 < v70 )
        {
          v4 = sub_CB6200(v4, (unsigned __int8 *)v71, v70);
          v74 = *(_BYTE **)(v4 + 32);
LABEL_107:
          if ( *(_BYTE **)(v4 + 24) == v74 )
          {
            v79 = sub_CB6200(v4, (unsigned __int8 *)"\n", 1u);
            v75 = *(_QWORD *)(v79 + 32);
            v4 = v79;
          }
          else
          {
            *v74 = 10;
            v75 = *(_QWORD *)(v4 + 32) + 1LL;
            *(_QWORD *)(v4 + 32) = v75;
          }
          if ( (unsigned __int64)(*(_QWORD *)(v4 + 24) - v75) <= 0x10 )
          {
            v4 = sub_CB6200(v4, "  Element Count: ", 0x11u);
          }
          else
          {
            v76 = _mm_load_si128((const __m128i *)&xmmword_43665C0);
            *(_BYTE *)(v75 + 16) = 32;
            *(__m128i *)v75 = v76;
            *(_QWORD *)(v4 + 32) += 17LL;
          }
          goto LABEL_67;
        }
        if ( (unsigned int)v70 >= 8 )
        {
          v80 = (unsigned __int64)(v72 + 1) & 0xFFFFFFFFFFFFFFF8LL;
          *v72 = *(_QWORD *)v71;
          *(_QWORD *)((char *)v72 + v70 - 8) = *(_QWORD *)&v71[v70 - 8];
          v81 = (char *)v72 - v80;
          v82 = (char *)(v71 - v81);
          if ( (((_DWORD)v70 + (_DWORD)v81) & 0xFFFFFFF8) >= 8 )
          {
            v83 = (v70 + (_DWORD)v81) & 0xFFFFFFF8;
            v84 = 0;
            do
            {
              v85 = v84;
              v84 += 8;
              *(_QWORD *)(v80 + v85) = *(_QWORD *)&v82[v85];
            }
            while ( v84 < v83 );
          }
        }
        else
        {
          if ( (v70 & 4) != 0 )
          {
            *(_DWORD *)v72 = *(_DWORD *)v71;
            *(_DWORD *)((char *)v72 + (unsigned int)v70 - 4) = *(_DWORD *)&v71[(unsigned int)v70 - 4];
            v73 = *(_QWORD *)(v4 + 32);
            goto LABEL_106;
          }
          *(_BYTE *)v72 = *v71;
          if ( (v70 & 2) != 0 )
          {
            *(_WORD *)((char *)v72 + (unsigned int)v70 - 2) = *(_WORD *)&v71[(unsigned int)v70 - 2];
            v73 = *(_QWORD *)(v4 + 32);
            goto LABEL_106;
          }
        }
        v73 = *(_QWORD *)(v4 + 32);
LABEL_106:
        v74 = (_BYTE *)(v70 + v73);
        *(_QWORD *)(v4 + 32) = v74;
        goto LABEL_107;
      default:
        goto LABEL_157;
    }
  }
  v26 = *(__m128i **)(v4 + 32);
  if ( *(_QWORD *)(v4 + 24) - (_QWORD)v26 <= 0xFu )
  {
    v4 = sub_CB6200(v4, "  Sampler Type: ", 0x10u);
  }
  else
  {
    *v26 = _mm_load_si128((const __m128i *)&xmmword_4366570);
    *(_QWORD *)(v4 + 32) += 16LL;
  }
  v27 = sub_22A6D30(a1);
  if ( v27 != 1 )
  {
    v28 = 4;
    v29 = "Mono";
    if ( v27 == 2 )
      goto LABEL_36;
    if ( !v27 )
    {
      v28 = 7;
      v29 = "Default";
      goto LABEL_36;
    }
LABEL_157:
    BUG();
  }
  v28 = 10;
  v29 = "Comparison";
LABEL_36:
  v30 = *(_QWORD **)(v4 + 32);
  if ( *(_QWORD *)(v4 + 24) - (_QWORD)v30 < v28 )
  {
    v4 = sub_CB6200(v4, (unsigned __int8 *)v29, v28);
    v25 = *(_BYTE **)(v4 + 32);
  }
  else
  {
    if ( (unsigned int)v28 >= 8 )
    {
      v64 = (unsigned __int64)(v30 + 1) & 0xFFFFFFFFFFFFFFF8LL;
      *v30 = *(_QWORD *)v29;
      *(_QWORD *)((char *)v30 + v28 - 8) = *(_QWORD *)&v29[v28 - 8];
      v65 = (char *)v30 - v64;
      v66 = (char *)(v29 - v65);
      if ( (((_DWORD)v28 + (_DWORD)v65) & 0xFFFFFFF8) >= 8 )
      {
        v67 = (v28 + (_DWORD)v65) & 0xFFFFFFF8;
        v68 = 0;
        do
        {
          v69 = v68;
          v68 += 8;
          *(_QWORD *)(v64 + v69) = *(_QWORD *)&v66[v69];
        }
        while ( v68 < v67 );
      }
      v86 = *(_QWORD *)(v4 + 32);
    }
    else
    {
      *(_DWORD *)v30 = *(_DWORD *)v29;
      *(_DWORD *)((char *)v30 + (unsigned int)v28 - 4) = *(_DWORD *)&v29[(unsigned int)v28 - 4];
      v86 = *(_QWORD *)(v4 + 32);
    }
    v25 = (_BYTE *)(v28 + v86);
    *(_QWORD *)(v4 + 32) = v25;
  }
  if ( *(_BYTE **)(v4 + 24) != v25 )
  {
LABEL_93:
    *v25 = 10;
    ++*(_QWORD *)(v4 + 32);
    return (char)v25;
  }
LABEL_56:
  v24 = v4;
LABEL_57:
  LOBYTE(v25) = sub_CB6200(v24, (unsigned __int8 *)"\n", 1u);
  return (char)v25;
}
