// Function: sub_2E37380
// Address: 0x2e37380
//
unsigned __int64 __fastcall sub_2E37380(__int64 a1, __int64 a2, char a3, __int64 a4)
{
  __int64 v8; // rdx
  __int64 v9; // rdi
  int v10; // r15d
  unsigned __int64 result; // rax
  __int64 v12; // r13
  unsigned __int64 v13; // rax
  _WORD *v14; // rdx
  __int64 v15; // r8
  size_t v16; // rdx
  unsigned __int8 *v17; // rsi
  void *v18; // rdi
  _WORD *v19; // rdi
  char *v20; // rsi
  __m128i *v21; // rdx
  __m128i v22; // xmm0
  char *v23; // rsi
  __int64 v24; // rdx
  __m128i v25; // xmm0
  void *v26; // rdx
  __int64 v27; // r13
  int v28; // r13d
  char *v29; // rsi
  void *v30; // rdx
  char *v31; // rsi
  __int64 v32; // rdx
  __int64 v33; // rdi
  char *v34; // rsi
  void *v35; // rdx
  char *v36; // rsi
  __m128i *v37; // rdx
  __m128i v38; // xmm0
  char *v39; // rsi
  void *v40; // rdx
  int v41; // eax
  char *v42; // rsi
  __int64 v43; // rdx
  __int64 v44; // rdi
  char *v45; // rsi
  __m128i *v46; // rdx
  __int64 v47; // rdi
  void *v48; // rdx
  int v49; // r13d
  __m128i *v50; // rdx
  __m128i si128; // xmm0
  __int64 v52; // rdx
  _DWORD *v53; // rdx
  __m128i *v54; // rdx
  __m128i v55; // xmm0
  unsigned __int8 *v56; // rax
  size_t v57; // rdx
  __int64 v58; // rax
  __int64 v59; // rsi
  _BYTE *v60; // rax
  __int64 v61; // rdi
  unsigned __int8 *v62; // rax
  size_t v63; // rdx
  __int64 v64; // rax
  __int64 v65; // rsi
  __int64 v66; // [rsp+0h] [rbp-B0h]
  __int64 v67; // [rsp+8h] [rbp-A8h]
  size_t v68; // [rsp+8h] [rbp-A8h]
  size_t v69; // [rsp+8h] [rbp-A8h]
  size_t v70; // [rsp+8h] [rbp-A8h]
  _QWORD v71[20]; // [rsp+10h] [rbp-A0h] BYREF

  v8 = *(_QWORD *)(a2 + 32);
  if ( (unsigned __int64)(*(_QWORD *)(a2 + 24) - v8) <= 2 )
  {
    v9 = sub_CB6200(a2, (unsigned __int8 *)"bb.", 3u);
  }
  else
  {
    *(_BYTE *)(v8 + 2) = 46;
    v9 = a2;
    *(_WORD *)v8 = 25186;
    *(_QWORD *)(a2 + 32) += 3LL;
  }
  v10 = a3 & 2;
  result = sub_CB59F0(v9, *(int *)(a1 + 24));
  if ( (a3 & 1) == 0 )
    goto LABEL_11;
  v12 = *(_QWORD *)(a1 + 16);
  if ( !v12 )
    goto LABEL_11;
  v13 = *(_QWORD *)(a2 + 24);
  v14 = *(_WORD **)(a2 + 32);
  if ( (*(_BYTE *)(v12 + 7) & 0x10) == 0 )
  {
    if ( v13 - (unsigned __int64)v14 <= 1 )
    {
      sub_CB6200(a2, (unsigned __int8 *)" (", 2u);
      v48 = *(void **)(a2 + 32);
    }
    else
    {
      *v14 = 10272;
      v48 = (void *)(*(_QWORD *)(a2 + 32) + 2LL);
      *(_QWORD *)(a2 + 32) = v48;
    }
    if ( *(_QWORD *)(a2 + 24) - (_QWORD)v48 <= 9u )
    {
      sub_CB6200(a2, "%ir-block.", 0xAu);
    }
    else
    {
      qmemcpy(v48, "%ir-block.", 10);
      *(_QWORD *)(a2 + 32) += 10LL;
    }
    if ( (*(_BYTE *)(v12 + 7) & 0x10) != 0 )
    {
      v62 = (unsigned __int8 *)sub_BD5D20(v12);
      v19 = *(_WORD **)(a2 + 32);
      if ( v63 > *(_QWORD *)(a2 + 24) - (_QWORD)v19 )
      {
        sub_CB6200(a2, v62, v63);
        v19 = *(_WORD **)(a2 + 32);
      }
      else if ( v63 )
      {
        v70 = v63;
        memcpy(v19, v62, v63);
        *(_QWORD *)(a2 + 32) += v70;
        v19 = *(_WORD **)(a2 + 32);
      }
      if ( !v10 )
        goto LABEL_94;
      if ( *(_BYTE *)(a1 + 217) )
        goto LABEL_111;
      if ( *(_QWORD *)(a1 + 224) )
        goto LABEL_20;
      if ( *(_BYTE *)(a1 + 216) )
        goto LABEL_117;
      goto LABEL_159;
    }
    if ( a4 )
    {
      v49 = sub_A5A720(a4, v12);
    }
    else
    {
      if ( !*(_QWORD *)(v12 + 72) )
        goto LABEL_106;
      v64 = sub_AA4B30(v12);
      sub_A558A0((__int64)v71, v64, 0);
      sub_A564B0((__int64)v71, *(_QWORD *)(v12 + 72));
      v65 = v12;
      v49 = sub_A5A720((__int64)v71, v12);
      sub_A55520(v71, v65);
    }
    if ( v49 != -1 )
    {
      sub_CB59F0(a2, v49);
LABEL_108:
      if ( !v10 )
        goto LABEL_93;
      if ( *(_BYTE *)(a1 + 217) )
      {
        v19 = *(_WORD **)(a2 + 32);
LABEL_111:
        v20 = ", ";
        goto LABEL_14;
      }
      if ( *(_QWORD *)(a1 + 224) )
      {
LABEL_19:
        v19 = *(_WORD **)(a2 + 32);
LABEL_20:
        v23 = ", ";
        goto LABEL_21;
      }
      v19 = *(_WORD **)(a2 + 32);
      if ( *(_BYTE *)(a1 + 216) )
        goto LABEL_117;
LABEL_159:
      if ( !*(_BYTE *)(a1 + 262) )
      {
        if ( !*(_BYTE *)(a1 + 235) )
        {
          if ( !*(_BYTE *)(a1 + 208) )
          {
            if ( !*(_QWORD *)(a1 + 252) )
              goto LABEL_120;
            goto LABEL_71;
          }
          goto LABEL_43;
        }
        goto LABEL_36;
      }
      goto LABEL_62;
    }
LABEL_106:
    v50 = *(__m128i **)(a2 + 32);
    if ( *(_QWORD *)(a2 + 24) - (_QWORD)v50 <= 0x10u )
    {
      sub_CB6200(a2, "<ir-block badref>", 0x11u);
    }
    else
    {
      si128 = _mm_load_si128((const __m128i *)&xmmword_42E9C60);
      v50[1].m128i_i8[0] = 62;
      *v50 = si128;
      *(_QWORD *)(a2 + 32) += 17LL;
    }
    goto LABEL_108;
  }
  if ( (unsigned __int64)v14 >= v13 )
  {
    v15 = sub_CB5D20(a2, 46);
  }
  else
  {
    v15 = a2;
    *(_QWORD *)(a2 + 32) = (char *)v14 + 1;
    *(_BYTE *)v14 = 46;
  }
  v67 = v15;
  v17 = (unsigned __int8 *)sub_BD5D20(v12);
  v18 = *(void **)(v67 + 32);
  result = *(_QWORD *)(v67 + 24) - (_QWORD)v18;
  if ( result < v16 )
  {
    result = sub_CB6200(v67, v17, v16);
  }
  else if ( v16 )
  {
    v66 = v67;
    v68 = v16;
    result = (unsigned __int64)memcpy(v18, v17, v16);
    *(_QWORD *)(v66 + 32) += v68;
  }
LABEL_11:
  if ( !v10 )
    return result;
  if ( *(_BYTE *)(a1 + 217) )
  {
    v19 = *(_WORD **)(a2 + 32);
    v20 = " (";
LABEL_14:
    if ( *(_QWORD *)(a2 + 24) - (_QWORD)v19 > 1u )
    {
      *v19 = *(_WORD *)v20;
      v21 = (__m128i *)(*(_QWORD *)(a2 + 32) + 2LL);
      *(_QWORD *)(a2 + 32) = v21;
    }
    else
    {
      sub_CB6200(a2, (unsigned __int8 *)v20, 2u);
      v21 = *(__m128i **)(a2 + 32);
    }
    if ( *(_QWORD *)(a2 + 24) - (_QWORD)v21 <= 0x1Au )
    {
      sub_CB6200(a2, "machine-block-address-taken", 0x1Bu);
    }
    else
    {
      v22 = _mm_load_si128((const __m128i *)&xmmword_444FF50);
      qmemcpy(&v21[1], "dress-taken", 11);
      *v21 = v22;
      *(_QWORD *)(a2 + 32) += 27LL;
    }
    if ( *(_QWORD *)(a1 + 224) )
      goto LABEL_19;
    if ( *(_BYTE *)(a1 + 216) )
    {
LABEL_116:
      v19 = *(_WORD **)(a2 + 32);
LABEL_117:
      v34 = ", ";
      goto LABEL_56;
    }
    if ( *(_BYTE *)(a1 + 262) )
    {
LABEL_61:
      v19 = *(_WORD **)(a2 + 32);
LABEL_62:
      v36 = ", ";
LABEL_63:
      if ( *(_QWORD *)(a2 + 24) - (_QWORD)v19 > 1u )
      {
        *v19 = *(_WORD *)v36;
        v37 = (__m128i *)(*(_QWORD *)(a2 + 32) + 2LL);
        *(_QWORD *)(a2 + 32) = v37;
      }
      else
      {
        sub_CB6200(a2, (unsigned __int8 *)v36, 2u);
        v37 = *(__m128i **)(a2 + 32);
      }
      if ( *(_QWORD *)(a2 + 24) - (_QWORD)v37 <= 0x1Bu )
      {
        sub_CB6200(a2, "inlineasm-br-indirect-target", 0x1Cu);
      }
      else
      {
        v38 = _mm_load_si128((const __m128i *)&xmmword_444FF70);
        qmemcpy(&v37[1], "irect-target", 12);
        *v37 = v38;
        *(_QWORD *)(a2 + 32) += 28LL;
      }
      goto LABEL_67;
    }
    if ( *(_BYTE *)(a1 + 235) )
      goto LABEL_35;
    if ( *(_BYTE *)(a1 + 208) )
      goto LABEL_42;
    if ( *(_QWORD *)(a1 + 252) )
      goto LABEL_70;
    if ( *(_BYTE *)(a1 + 248) )
    {
      v19 = *(_WORD **)(a2 + 32);
LABEL_79:
      v42 = ", ";
LABEL_80:
      if ( *(_QWORD *)(a2 + 24) - (_QWORD)v19 > 1u )
      {
        *v19 = *(_WORD *)v42;
        v43 = *(_QWORD *)(a2 + 32) + 2LL;
        *(_QWORD *)(a2 + 32) = v43;
      }
      else
      {
        sub_CB6200(a2, (unsigned __int8 *)v42, 2u);
        v43 = *(_QWORD *)(a2 + 32);
      }
      if ( (unsigned __int64)(*(_QWORD *)(a2 + 24) - v43) <= 5 )
      {
        v44 = sub_CB6200(a2, "bb_id ", 6u);
      }
      else
      {
        *(_DWORD *)v43 = 1767858786;
        *(_WORD *)(v43 + 4) = 8292;
        v44 = a2;
        *(_QWORD *)(a2 + 32) += 6LL;
      }
      sub_CB59D0(v44, *(unsigned int *)(a1 + 240));
      if ( *(_DWORD *)(a1 + 244) )
      {
        v60 = *(_BYTE **)(a2 + 32);
        if ( *(_BYTE **)(a2 + 24) == v60 )
        {
          v61 = sub_CB6200(a2, (unsigned __int8 *)" ", 1u);
        }
        else
        {
          *v60 = 32;
          v61 = a2;
          ++*(_QWORD *)(a2 + 32);
        }
        sub_CB59D0(v61, *(unsigned int *)(a1 + 244));
        v19 = *(_WORD **)(a2 + 32);
        if ( !*(_DWORD *)(a1 + 28) )
          goto LABEL_94;
        goto LABEL_87;
      }
      v19 = *(_WORD **)(a2 + 32);
LABEL_86:
      if ( !*(_DWORD *)(a1 + 28) )
        goto LABEL_94;
      goto LABEL_87;
    }
    if ( *(_DWORD *)(a1 + 28) )
    {
      v19 = *(_WORD **)(a2 + 32);
      goto LABEL_87;
    }
LABEL_93:
    v19 = *(_WORD **)(a2 + 32);
    goto LABEL_94;
  }
  if ( *(_QWORD *)(a1 + 224) )
  {
    v19 = *(_WORD **)(a2 + 32);
    v23 = " (";
LABEL_21:
    if ( *(_QWORD *)(a2 + 24) - (_QWORD)v19 > 1u )
    {
      *v19 = *(_WORD *)v23;
      v24 = *(_QWORD *)(a2 + 32) + 2LL;
      *(_QWORD *)(a2 + 32) = v24;
    }
    else
    {
      sub_CB6200(a2, (unsigned __int8 *)v23, 2u);
      v24 = *(_QWORD *)(a2 + 32);
    }
    if ( (unsigned __int64)(*(_QWORD *)(a2 + 24) - v24) <= 0x16 )
    {
      sub_CB6200(a2, "ir-block-address-taken ", 0x17u);
      v26 = *(void **)(a2 + 32);
    }
    else
    {
      v25 = _mm_load_si128((const __m128i *)&xmmword_444FF60);
      *(_DWORD *)(v24 + 16) = 1801548845;
      *(_WORD *)(v24 + 20) = 28261;
      *(_BYTE *)(v24 + 22) = 32;
      *(__m128i *)v24 = v25;
      v26 = (void *)(*(_QWORD *)(a2 + 32) + 23LL);
      *(_QWORD *)(a2 + 32) = v26;
    }
    v27 = *(_QWORD *)(a1 + 224);
    if ( *(_QWORD *)(a2 + 24) - (_QWORD)v26 <= 9u )
    {
      sub_CB6200(a2, "%ir-block.", 0xAu);
    }
    else
    {
      qmemcpy(v26, "%ir-block.", 10);
      *(_QWORD *)(a2 + 32) += 10LL;
    }
    if ( (*(_BYTE *)(v27 + 7) & 0x10) != 0 )
    {
      v56 = (unsigned __int8 *)sub_BD5D20(v27);
      v19 = *(_WORD **)(a2 + 32);
      if ( *(_QWORD *)(a2 + 24) - (_QWORD)v19 < v57 )
      {
        sub_CB6200(a2, v56, v57);
        v19 = *(_WORD **)(a2 + 32);
      }
      else if ( v57 )
      {
        v69 = v57;
        memcpy(v19, v56, v57);
        *(_QWORD *)(a2 + 32) += v69;
        v19 = *(_WORD **)(a2 + 32);
      }
      if ( !*(_BYTE *)(a1 + 216) )
      {
        if ( !*(_BYTE *)(a1 + 262) )
        {
          if ( *(_BYTE *)(a1 + 235) )
            goto LABEL_36;
          if ( *(_BYTE *)(a1 + 208) )
            goto LABEL_43;
          goto LABEL_186;
        }
        goto LABEL_62;
      }
      goto LABEL_117;
    }
    if ( a4 )
    {
      v28 = sub_A5A720(a4, v27);
    }
    else
    {
      if ( !*(_QWORD *)(v27 + 72) )
        goto LABEL_125;
      v58 = sub_AA4B30(v27);
      sub_A558A0((__int64)v71, v58, 0);
      sub_A564B0((__int64)v71, *(_QWORD *)(v27 + 72));
      v59 = v27;
      v28 = sub_A5A720((__int64)v71, v27);
      sub_A55520(v71, v59);
    }
    if ( v28 != -1 )
    {
      sub_CB59F0(a2, v28);
      goto LABEL_32;
    }
LABEL_125:
    v54 = *(__m128i **)(a2 + 32);
    if ( *(_QWORD *)(a2 + 24) - (_QWORD)v54 <= 0x10u )
    {
      sub_CB6200(a2, "<ir-block badref>", 0x11u);
    }
    else
    {
      v55 = _mm_load_si128((const __m128i *)&xmmword_42E9C60);
      v54[1].m128i_i8[0] = 62;
      *v54 = v55;
      *(_QWORD *)(a2 + 32) += 17LL;
    }
LABEL_32:
    if ( !*(_BYTE *)(a1 + 216) )
    {
      if ( !*(_BYTE *)(a1 + 262) )
      {
        if ( *(_BYTE *)(a1 + 235) )
        {
LABEL_35:
          v19 = *(_WORD **)(a2 + 32);
LABEL_36:
          v29 = ", ";
LABEL_37:
          if ( *(_QWORD *)(a2 + 24) - (_QWORD)v19 > 1u )
          {
            *v19 = *(_WORD *)v29;
            v30 = (void *)(*(_QWORD *)(a2 + 32) + 2LL);
            *(_QWORD *)(a2 + 32) = v30;
          }
          else
          {
            sub_CB6200(a2, (unsigned __int8 *)v29, 2u);
            v30 = *(void **)(a2 + 32);
          }
          if ( *(_QWORD *)(a2 + 24) - (_QWORD)v30 <= 0xEu )
          {
            sub_CB6200(a2, "ehfunclet-entry", 0xFu);
          }
          else
          {
            qmemcpy(v30, "ehfunclet-entry", 15);
            *(_QWORD *)(a2 + 32) += 15LL;
          }
          if ( !*(_BYTE *)(a1 + 208) )
          {
LABEL_49:
            if ( !*(_QWORD *)(a1 + 252) )
              goto LABEL_50;
            goto LABEL_70;
          }
          goto LABEL_42;
        }
        v19 = *(_WORD **)(a2 + 32);
        if ( *(_BYTE *)(a1 + 208) )
          goto LABEL_43;
LABEL_186:
        if ( !*(_QWORD *)(a1 + 252) )
          goto LABEL_120;
        goto LABEL_71;
      }
      goto LABEL_61;
    }
    goto LABEL_116;
  }
  if ( !*(_BYTE *)(a1 + 216) )
  {
    if ( !*(_BYTE *)(a1 + 262) )
    {
      if ( !*(_BYTE *)(a1 + 235) )
      {
        if ( !*(_BYTE *)(a1 + 208) )
        {
          result = (unsigned int)(*(_DWORD *)(a1 + 252) | *(_DWORD *)(a1 + 256));
          if ( !*(_QWORD *)(a1 + 252) )
          {
            if ( !*(_BYTE *)(a1 + 248) )
            {
              if ( !*(_DWORD *)(a1 + 28) )
                return result;
              v19 = *(_WORD **)(a2 + 32);
              v45 = " (";
              goto LABEL_88;
            }
            v19 = *(_WORD **)(a2 + 32);
            v42 = " (";
            goto LABEL_80;
          }
          v19 = *(_WORD **)(a2 + 32);
          v39 = " (";
LABEL_72:
          if ( *(_QWORD *)(a2 + 24) - (_QWORD)v19 > 1u )
          {
            *v19 = *(_WORD *)v39;
            v40 = (void *)(*(_QWORD *)(a2 + 32) + 2LL);
            *(_QWORD *)(a2 + 32) = v40;
          }
          else
          {
            sub_CB6200(a2, (unsigned __int8 *)v39, 2u);
            v40 = *(void **)(a2 + 32);
          }
          if ( *(_QWORD *)(a2 + 24) - (_QWORD)v40 <= 0xAu )
          {
            sub_CB6200(a2, "bbsections ", 0xBu);
          }
          else
          {
            qmemcpy(v40, "bbsections ", 11);
            *(_QWORD *)(a2 + 32) += 11LL;
          }
          v41 = *(_DWORD *)(a1 + 252);
          if ( v41 == 1 )
          {
            v52 = *(_QWORD *)(a2 + 32);
            if ( (unsigned __int64)(*(_QWORD *)(a2 + 24) - v52) <= 8 )
            {
              sub_CB6200(a2, (unsigned __int8 *)"Exception", 9u);
              v19 = *(_WORD **)(a2 + 32);
            }
            else
            {
              *(_BYTE *)(v52 + 8) = 110;
              *(_QWORD *)v52 = 0x6F69747065637845LL;
              v19 = (_WORD *)(*(_QWORD *)(a2 + 32) + 9LL);
              *(_QWORD *)(a2 + 32) = v19;
            }
          }
          else
          {
            if ( v41 != 2 )
            {
              sub_CB59D0(a2, *(unsigned int *)(a1 + 256));
              v19 = *(_WORD **)(a2 + 32);
              if ( !*(_BYTE *)(a1 + 248) )
                goto LABEL_86;
              goto LABEL_79;
            }
            v53 = *(_DWORD **)(a2 + 32);
            if ( *(_QWORD *)(a2 + 24) - (_QWORD)v53 <= 3u )
            {
              sub_CB6200(a2, (unsigned __int8 *)"Cold", 4u);
              v19 = *(_WORD **)(a2 + 32);
            }
            else
            {
              *v53 = 1684827971;
              v19 = (_WORD *)(*(_QWORD *)(a2 + 32) + 4LL);
              *(_QWORD *)(a2 + 32) = v19;
            }
          }
LABEL_120:
          if ( !*(_BYTE *)(a1 + 248) )
            goto LABEL_86;
          goto LABEL_79;
        }
        v19 = *(_WORD **)(a2 + 32);
        v31 = " (";
        goto LABEL_44;
      }
      v19 = *(_WORD **)(a2 + 32);
      v29 = " (";
      goto LABEL_37;
    }
    v19 = *(_WORD **)(a2 + 32);
    v36 = " (";
    goto LABEL_63;
  }
  v19 = *(_WORD **)(a2 + 32);
  v34 = " (";
LABEL_56:
  if ( *(_QWORD *)(a2 + 24) - (_QWORD)v19 > 1u )
  {
    *v19 = *(_WORD *)v34;
    v35 = (void *)(*(_QWORD *)(a2 + 32) + 2LL);
    *(_QWORD *)(a2 + 32) = v35;
  }
  else
  {
    sub_CB6200(a2, (unsigned __int8 *)v34, 2u);
    v35 = *(void **)(a2 + 32);
  }
  if ( *(_QWORD *)(a2 + 24) - (_QWORD)v35 <= 0xAu )
  {
    sub_CB6200(a2, "landing-pad", 0xBu);
  }
  else
  {
    qmemcpy(v35, "landing-pad", 11);
    *(_QWORD *)(a2 + 32) += 11LL;
  }
  if ( *(_BYTE *)(a1 + 262) )
    goto LABEL_61;
LABEL_67:
  if ( *(_BYTE *)(a1 + 235) )
    goto LABEL_35;
  if ( *(_BYTE *)(a1 + 208) )
  {
LABEL_42:
    v19 = *(_WORD **)(a2 + 32);
LABEL_43:
    v31 = ", ";
LABEL_44:
    if ( *(_QWORD *)(a2 + 24) - (_QWORD)v19 > 1u )
    {
      *v19 = *(_WORD *)v31;
      v32 = *(_QWORD *)(a2 + 32) + 2LL;
      *(_QWORD *)(a2 + 32) = v32;
    }
    else
    {
      sub_CB6200(a2, (unsigned __int8 *)v31, 2u);
      v32 = *(_QWORD *)(a2 + 32);
    }
    if ( (unsigned __int64)(*(_QWORD *)(a2 + 24) - v32) <= 5 )
    {
      v33 = sub_CB6200(a2, (unsigned __int8 *)"align ", 6u);
    }
    else
    {
      *(_DWORD *)v32 = 1734962273;
      v33 = a2;
      *(_WORD *)(v32 + 4) = 8302;
      *(_QWORD *)(a2 + 32) += 6LL;
    }
    sub_CB59D0(v33, 1LL << *(_BYTE *)(a1 + 208));
    goto LABEL_49;
  }
  if ( *(_QWORD *)(a1 + 252) )
  {
LABEL_70:
    v19 = *(_WORD **)(a2 + 32);
LABEL_71:
    v39 = ", ";
    goto LABEL_72;
  }
LABEL_50:
  if ( *(_BYTE *)(a1 + 248) )
  {
    v19 = *(_WORD **)(a2 + 32);
    v42 = ", ";
    goto LABEL_80;
  }
  v19 = *(_WORD **)(a2 + 32);
  if ( *(_DWORD *)(a1 + 28) )
  {
LABEL_87:
    v45 = ", ";
LABEL_88:
    if ( *(_QWORD *)(a2 + 24) - (_QWORD)v19 > 1u )
    {
      *v19 = *(_WORD *)v45;
      v46 = (__m128i *)(*(_QWORD *)(a2 + 32) + 2LL);
      *(_QWORD *)(a2 + 32) = v46;
    }
    else
    {
      sub_CB6200(a2, (unsigned __int8 *)v45, 2u);
      v46 = *(__m128i **)(a2 + 32);
    }
    if ( *(_QWORD *)(a2 + 24) - (_QWORD)v46 <= 0xFu )
    {
      v47 = sub_CB6200(a2, "call-frame-size ", 0x10u);
    }
    else
    {
      v47 = a2;
      *v46 = _mm_load_si128((const __m128i *)&xmmword_444FF80);
      *(_QWORD *)(a2 + 32) += 16LL;
    }
    sub_CB59D0(v47, *(unsigned int *)(a1 + 28));
    goto LABEL_93;
  }
LABEL_94:
  if ( (unsigned __int64)v19 >= *(_QWORD *)(a2 + 24) )
    return sub_CB5D20(a2, 41);
  result = (unsigned __int64)v19 + 1;
  *(_QWORD *)(a2 + 32) = (char *)v19 + 1;
  *(_BYTE *)v19 = 41;
  return result;
}
