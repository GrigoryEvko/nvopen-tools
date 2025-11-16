// Function: sub_30AF240
// Address: 0x30af240
//
__int64 __fastcall sub_30AF240(__int64 a1, __int64 *a2, __int64 a3, __int64 a4)
{
  __int64 **v7; // r14
  __int64 v8; // rax
  __int64 v9; // r12
  __int64 v10; // r15
  __m128i *v11; // rdx
  __m128i si128; // xmm0
  const char *v13; // rax
  size_t v14; // rdx
  _BYTE *v15; // rdi
  unsigned __int8 *v16; // rsi
  unsigned __int64 v17; // rax
  __int64 v18; // rbx
  __int64 v19; // rdi
  __m128i *v20; // rdx
  __m128i v21; // xmm0
  __int64 v22; // r8
  __m128i *v23; // rdx
  __m128i v24; // xmm0
  _BYTE *v25; // rax
  __int64 v26; // rdi
  __int64 v27; // r12
  void *v28; // rdx
  unsigned int v29; // esi
  signed __int64 v30; // r8
  int v31; // edx
  __int64 v32; // rdi
  void *v33; // rdx
  __int64 v34; // rdi
  void *v35; // rdx
  __int64 v36; // rax
  __int64 v37; // rdx
  __int64 v38; // rax
  __int64 v39; // rdx
  __int64 v40; // rax
  __int64 v41; // rdx
  __int64 v42; // rdx
  __int64 v43; // r8
  __int64 v44; // rdx
  __int64 v45; // r8
  void *v46; // rdx
  __int64 v47; // r8
  __int64 v48; // rdx
  __int64 v49; // r8
  __int64 v50; // rdx
  __int64 v51; // rdx
  __int64 v53; // rax
  unsigned __int64 v54; // rax
  __int64 v55; // rax
  __int64 v57; // [rsp+10h] [rbp-90h]
  __int64 v58; // [rsp+18h] [rbp-88h]
  __int64 v59; // [rsp+20h] [rbp-80h]
  __int64 v60; // [rsp+28h] [rbp-78h]
  __int64 v61; // [rsp+28h] [rbp-78h]
  __int64 v62; // [rsp+28h] [rbp-78h]
  __int64 v63; // [rsp+28h] [rbp-78h]
  signed __int64 v64; // [rsp+28h] [rbp-78h]
  size_t v65; // [rsp+28h] [rbp-78h]
  __int64 v66; // [rsp+30h] [rbp-70h] BYREF
  __int64 v67; // [rsp+38h] [rbp-68h]
  __int64 v68; // [rsp+40h] [rbp-60h] BYREF
  __int64 v69; // [rsp+48h] [rbp-58h]
  __int64 v70; // [rsp+50h] [rbp-50h] BYREF
  __int64 v71; // [rsp+58h] [rbp-48h]
  __int64 v72; // [rsp+60h] [rbp-40h] BYREF
  __int64 v73; // [rsp+68h] [rbp-38h]

  v7 = (__int64 **)(sub_BC1CD0(a4, &unk_4F89C30, a3) + 8);
  v8 = sub_BC1CD0(a4, &unk_4F6D3F8, a3);
  v9 = *a2;
  v10 = v8 + 8;
  v11 = *(__m128i **)(*a2 + 32);
  if ( *(_QWORD *)(*a2 + 24) - (_QWORD)v11 <= 0x35u )
  {
    v9 = sub_CB6200(*a2, "Printing analysis 'Cost Model Analysis' for function '", 0x36u);
  }
  else
  {
    si128 = _mm_load_si128((const __m128i *)&xmmword_3F70AE0);
    v11[3].m128i_i32[0] = 1852795252;
    v11[3].m128i_i16[2] = 10016;
    *v11 = si128;
    v11[1] = _mm_load_si128((const __m128i *)&xmmword_44CB940);
    v11[2] = _mm_load_si128((const __m128i *)&xmmword_4366110);
    *(_QWORD *)(v9 + 32) += 54LL;
  }
  v13 = sub_BD5D20(a3);
  v15 = *(_BYTE **)(v9 + 32);
  v16 = (unsigned __int8 *)v13;
  v17 = *(_QWORD *)(v9 + 24) - (_QWORD)v15;
  if ( v17 < v14 )
  {
    v55 = sub_CB6200(v9, v16, v14);
    v15 = *(_BYTE **)(v55 + 32);
    v9 = v55;
    v17 = *(_QWORD *)(v55 + 24) - (_QWORD)v15;
  }
  else if ( v14 )
  {
    v65 = v14;
    memcpy(v15, v16, v14);
    v15 = (_BYTE *)(v65 + *(_QWORD *)(v9 + 32));
    v54 = *(_QWORD *)(v9 + 24) - (_QWORD)v15;
    *(_QWORD *)(v9 + 32) = v15;
    if ( v54 > 2 )
      goto LABEL_6;
    goto LABEL_60;
  }
  if ( v17 > 2 )
  {
LABEL_6:
    v15[2] = 10;
    *(_WORD *)v15 = 14887;
    *(_QWORD *)(v9 + 32) += 3LL;
    goto LABEL_7;
  }
LABEL_60:
  sub_CB6200(v9, "':\n", 3u);
LABEL_7:
  v57 = a3 + 72;
  v58 = *(_QWORD *)(a3 + 80);
  if ( v58 != a3 + 72 )
  {
    while ( 1 )
    {
      if ( !v58 )
        BUG();
      v18 = *(_QWORD *)(v58 + 32);
      v59 = v58 + 24;
      if ( v18 != v58 + 24 )
        break;
LABEL_42:
      v58 = *(_QWORD *)(v58 + 8);
      if ( v57 == v58 )
        goto LABEL_43;
    }
    while ( 1 )
    {
      v26 = *a2;
      v27 = v18 - 24;
      if ( !v18 )
        v27 = 0;
      v28 = *(void **)(v26 + 32);
      if ( *(_QWORD *)(v26 + 24) - (_QWORD)v28 <= 0xBu )
      {
        sub_CB6200(v26, "Cost Model: ", 0xCu);
        v29 = dword_502E808;
        if ( dword_502E808 != 4 )
        {
LABEL_22:
          if ( v29 > 3 )
            BUG();
          v30 = sub_30AE990(v27, v29, v7, v10);
          if ( v31 )
          {
            v32 = *a2;
            v33 = *(void **)(*a2 + 32);
            if ( *(_QWORD *)(*a2 + 24) - (_QWORD)v33 <= 0xBu )
            {
              sub_CB6200(v32, "Invalid cost", 0xCu);
            }
            else
            {
              qmemcpy(v33, "Invalid cost", 12);
              *(_QWORD *)(v32 + 32) += 12LL;
            }
          }
          else
          {
            v19 = *a2;
            v20 = *(__m128i **)(*a2 + 32);
            if ( *(_QWORD *)(*a2 + 24) - (_QWORD)v20 <= 0x1Au )
            {
              v64 = v30;
              v53 = sub_CB6200(v19, "Found an estimated cost of ", 0x1Bu);
              v30 = v64;
              v19 = v53;
            }
            else
            {
              v21 = _mm_load_si128((const __m128i *)&xmmword_44CB950);
              qmemcpy(&v20[1], "ed cost of ", 11);
              *v20 = v21;
              *(_QWORD *)(v19 + 32) += 27LL;
            }
            sub_CB59F0(v19, v30);
          }
          v22 = *a2;
          v23 = *(__m128i **)(*a2 + 32);
          if ( *(_QWORD *)(*a2 + 24) - (_QWORD)v23 <= 0x11u )
          {
            v22 = sub_CB6200(*a2, " for instruction: ", 0x12u);
          }
          else
          {
            v24 = _mm_load_si128((const __m128i *)&xmmword_4289540);
            v23[1].m128i_i16[0] = 8250;
            *v23 = v24;
            *(_QWORD *)(v22 + 32) += 18LL;
          }
          goto LABEL_16;
        }
      }
      else
      {
        qmemcpy(v28, "Cost Model: ", 12);
        *(_QWORD *)(v26 + 32) += 12LL;
        v29 = dword_502E808;
        if ( dword_502E808 != 4 )
          goto LABEL_22;
      }
      v34 = *a2;
      v35 = *(void **)(*a2 + 32);
      if ( *(_QWORD *)(*a2 + 24) - (_QWORD)v35 <= 0xEu )
      {
        sub_CB6200(v34, "Found costs of ", 0xFu);
      }
      else
      {
        qmemcpy(v35, "Found costs of ", 15);
        *(_QWORD *)(v34 + 32) += 15LL;
      }
      v36 = sub_30AE990(v27, 0, v7, v10);
      v67 = v37;
      v66 = v36;
      v38 = sub_30AE990(v27, 2, v7, v10);
      v69 = v39;
      v68 = v38;
      v40 = sub_30AE990(v27, 1, v7, v10);
      v71 = v41;
      v70 = v40;
      v72 = sub_30AE990(v27, 3, v7, v10);
      v73 = v42;
      if ( (_DWORD)v67 == (_DWORD)v69
        && v66 == v68
        && (_DWORD)v67 == (_DWORD)v71
        && v66 == v70
        && (_DWORD)v67 == (_DWORD)v73
        && v66 == v72 )
      {
        sub_C68B50((__int64)&v66, *a2);
      }
      else
      {
        v43 = *a2;
        v44 = *(_QWORD *)(*a2 + 32);
        if ( (unsigned __int64)(*(_QWORD *)(*a2 + 24) - v44) <= 5 )
        {
          v43 = sub_CB6200(*a2, "RThru:", 6u);
        }
        else
        {
          *(_DWORD *)v44 = 1919439954;
          *(_WORD *)(v44 + 4) = 14965;
          *(_QWORD *)(v43 + 32) += 6LL;
        }
        v61 = v43;
        sub_C68B50((__int64)&v66, v43);
        v45 = v61;
        v46 = *(void **)(v61 + 32);
        if ( *(_QWORD *)(v61 + 24) - (_QWORD)v46 <= 9u )
        {
          v45 = sub_CB6200(v61, " CodeSize:", 0xAu);
        }
        else
        {
          qmemcpy(v46, " CodeSize:", 10);
          *(_QWORD *)(v61 + 32) += 10LL;
        }
        v62 = v45;
        sub_C68B50((__int64)&v68, v45);
        v47 = v62;
        v48 = *(_QWORD *)(v62 + 32);
        if ( (unsigned __int64)(*(_QWORD *)(v62 + 24) - v48) <= 4 )
        {
          v47 = sub_CB6200(v62, " Lat:", 5u);
        }
        else
        {
          *(_DWORD *)v48 = 1952533536;
          *(_BYTE *)(v48 + 4) = 58;
          *(_QWORD *)(v62 + 32) += 5LL;
        }
        v63 = v47;
        sub_C68B50((__int64)&v70, v47);
        v49 = v63;
        v50 = *(_QWORD *)(v63 + 32);
        if ( (unsigned __int64)(*(_QWORD *)(v63 + 24) - v50) <= 8 )
        {
          v49 = sub_CB6200(v63, " SizeLat:", 9u);
        }
        else
        {
          *(_BYTE *)(v50 + 8) = 58;
          *(_QWORD *)v50 = 0x74614C657A695320LL;
          *(_QWORD *)(v63 + 32) += 9LL;
        }
        sub_C68B50((__int64)&v72, v49);
      }
      v22 = *a2;
      v51 = *(_QWORD *)(*a2 + 32);
      if ( (unsigned __int64)(*(_QWORD *)(*a2 + 24) - v51) <= 5 )
      {
        v22 = sub_CB6200(*a2, (unsigned __int8 *)" for: ", 6u);
      }
      else
      {
        *(_DWORD *)v51 = 1919903264;
        *(_WORD *)(v51 + 4) = 8250;
        *(_QWORD *)(v22 + 32) += 6LL;
      }
LABEL_16:
      v60 = v22;
      sub_A69870(v27, (_BYTE *)v22, 0);
      v25 = *(_BYTE **)(v60 + 32);
      if ( *(_BYTE **)(v60 + 24) == v25 )
      {
        sub_CB6200(v60, (unsigned __int8 *)"\n", 1u);
        v18 = *(_QWORD *)(v18 + 8);
        if ( v59 == v18 )
          goto LABEL_42;
      }
      else
      {
        *v25 = 10;
        ++*(_QWORD *)(v60 + 32);
        v18 = *(_QWORD *)(v18 + 8);
        if ( v59 == v18 )
          goto LABEL_42;
      }
    }
  }
LABEL_43:
  *(_QWORD *)(a1 + 48) = 0;
  *(_QWORD *)(a1 + 8) = a1 + 32;
  *(_QWORD *)(a1 + 56) = a1 + 80;
  *(_QWORD *)(a1 + 16) = 0x100000002LL;
  *(_QWORD *)(a1 + 64) = 2;
  *(_QWORD *)(a1 + 32) = &qword_4F82400;
  *(_DWORD *)(a1 + 72) = 0;
  *(_BYTE *)(a1 + 76) = 1;
  *(_DWORD *)(a1 + 24) = 0;
  *(_BYTE *)(a1 + 28) = 1;
  *(_QWORD *)a1 = 1;
  return a1;
}
