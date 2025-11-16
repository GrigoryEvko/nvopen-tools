// Function: sub_2FE12A0
// Address: 0x2fe12a0
//
__m128i *__fastcall sub_2FE12A0(__m128i *a1, __int64 a2, __int64 a3, __int64 a4, unsigned int a5, __int64 a6)
{
  int v9; // eax
  __int64 v10; // rcx
  int v11; // ebx
  __int64 v13; // rbx
  __m128i *v14; // r9
  void *v15; // r10
  size_t v16; // rbx
  const __m128i *v17; // r13
  char *v18; // rdi
  unsigned __int64 v19; // rax
  __int64 v20; // rax
  size_t v21; // rdx
  char *v22; // r8
  char *v23; // rsi
  unsigned int v24; // r8d
  __int64 v25; // r11
  char *v26; // rax
  _QWORD *v27; // r10
  char *v28; // rsi
  size_t v29; // rax
  void *v30; // rdi
  char *v31; // rdx
  unsigned int v32; // r13d
  _QWORD *v33; // rdi
  char *v34; // rdx
  __int64 v35; // rax
  unsigned int v36; // r13d
  _QWORD *v37; // r10
  unsigned __int8 *v38; // rax
  size_t v39; // rdx
  void *v40; // rdi
  unsigned __int64 v41; // rdi
  char *v42; // rsi
  char *v43; // r8
  unsigned int v44; // esi
  unsigned int v45; // eax
  __int64 v46; // r9
  __int64 v47; // rax
  _QWORD *v48; // rdi
  __int64 v49; // rax
  __int64 v50; // rax
  int v51; // [rsp+8h] [rbp-E8h]
  int v52; // [rsp+10h] [rbp-E0h]
  int v53; // [rsp+10h] [rbp-E0h]
  int v54; // [rsp+10h] [rbp-E0h]
  int v55; // [rsp+18h] [rbp-D8h]
  unsigned int v56; // [rsp+18h] [rbp-D8h]
  int v57; // [rsp+18h] [rbp-D8h]
  __m128i *v59; // [rsp+20h] [rbp-D0h]
  size_t v60; // [rsp+20h] [rbp-D0h]
  unsigned int v61; // [rsp+20h] [rbp-D0h]
  __int64 v62; // [rsp+20h] [rbp-D0h]
  int v63; // [rsp+20h] [rbp-D0h]
  __m128i *v65; // [rsp+28h] [rbp-C8h]
  __m128i *v66; // [rsp+28h] [rbp-C8h]
  void *v67; // [rsp+28h] [rbp-C8h]
  __int64 v68; // [rsp+28h] [rbp-C8h]
  int v69; // [rsp+28h] [rbp-C8h]
  int v70; // [rsp+28h] [rbp-C8h]
  int v71; // [rsp+28h] [rbp-C8h]
  __int64 v72; // [rsp+28h] [rbp-C8h]
  size_t v73; // [rsp+28h] [rbp-C8h]
  unsigned int v74; // [rsp+28h] [rbp-C8h]
  unsigned int v75; // [rsp+28h] [rbp-C8h]
  int v76; // [rsp+28h] [rbp-C8h]
  __m128i v77; // [rsp+30h] [rbp-C0h] BYREF
  __m128i *v78; // [rsp+40h] [rbp-B0h] BYREF
  __m128i *v79; // [rsp+48h] [rbp-A8h]
  __m128i *v80; // [rsp+50h] [rbp-A0h]
  __m128i *v81; // [rsp+60h] [rbp-90h] BYREF
  __int64 v82; // [rsp+68h] [rbp-88h]
  __m128i v83; // [rsp+70h] [rbp-80h] BYREF
  _QWORD v84[3]; // [rsp+80h] [rbp-70h] BYREF
  char *v85; // [rsp+98h] [rbp-58h]
  char *v86; // [rsp+A0h] [rbp-50h]
  __int64 v87; // [rsp+A8h] [rbp-48h]
  __m128i **v88; // [rsp+B0h] [rbp-40h]

  if ( (unsigned int)*(unsigned __int16 *)(a3 + 68) - 1 > 1 )
  {
    a1->m128i_i64[0] = (__int64)a1[1].m128i_i64;
    sub_2FDCEA0(a1->m128i_i64, byte_3F871B3, (__int64)byte_3F871B3);
    return a1;
  }
  v83.m128i_i8[0] = 0;
  v81 = &v83;
  v87 = 0x100000000LL;
  v84[0] = &unk_49DD210;
  v82 = 0;
  v84[1] = 0;
  v84[2] = 0;
  v85 = 0;
  v86 = 0;
  v88 = &v81;
  sub_CB5980((__int64)v84, 0, 0, 0);
  if ( a5 == 1 )
  {
    v13 = *(_QWORD *)(a4 + 24);
    v14 = 0;
    v78 = 0;
    v79 = 0;
    v80 = 0;
    if ( (v13 & 1) != 0 )
    {
      v77.m128i_i64[1] = 10;
      v77.m128i_i64[0] = (__int64)"sideeffect";
      sub_C677B0((const __m128i **)&v78, 0, &v77);
      v14 = v79;
    }
    if ( (v13 & 8) != 0 )
    {
      v77.m128i_i64[1] = 7;
      v77.m128i_i64[0] = (__int64)"mayload";
      if ( v80 == v14 )
      {
        sub_C677B0((const __m128i **)&v78, v14, &v77);
        v14 = v79;
      }
      else
      {
        if ( v14 )
        {
          *v14 = _mm_load_si128(&v77);
          v14 = v79;
        }
        v79 = ++v14;
      }
    }
    if ( (v13 & 0x10) != 0 )
    {
      v77.m128i_i64[1] = 8;
      v77.m128i_i64[0] = (__int64)"maystore";
      if ( v80 == v14 )
      {
        sub_C677B0((const __m128i **)&v78, v14, &v77);
        v14 = v79;
      }
      else
      {
        if ( v14 )
        {
          *v14 = _mm_load_si128(&v77);
          v14 = v79;
        }
        v79 = ++v14;
      }
    }
    if ( (v13 & 0x20) != 0 )
    {
      v77.m128i_i64[1] = 12;
      v77.m128i_i64[0] = (__int64)"isconvergent";
      if ( v80 == v14 )
      {
        sub_C677B0((const __m128i **)&v78, v14, &v77);
        v14 = v79;
      }
      else
      {
        if ( v14 )
        {
          *v14 = _mm_load_si128(&v77);
          v14 = v79;
        }
        v79 = ++v14;
      }
    }
    if ( (v13 & 2) != 0 )
    {
      v77.m128i_i64[1] = 10;
      v77.m128i_i64[0] = (__int64)"alignstack";
      if ( v80 == v14 )
      {
        sub_C677B0((const __m128i **)&v78, v14, &v77);
        v14 = v79;
      }
      else
      {
        if ( v14 )
        {
          *v14 = _mm_load_si128(&v77);
          v14 = v79;
        }
        v79 = ++v14;
      }
    }
    if ( (v13 & 4) == 0 )
    {
      v77.m128i_i64[1] = 10;
      v77.m128i_i64[0] = (__int64)"attdialect";
      if ( v14 == v80 )
      {
        sub_C677B0((const __m128i **)&v78, v14, &v77);
        v14 = v79;
      }
      else
      {
        if ( v14 )
        {
          *v14 = _mm_load_si128(&v77);
          v14 = v79;
        }
        v79 = ++v14;
      }
    }
    if ( v78 != v14 )
    {
      v15 = (void *)v78->m128i_i64[0];
      v16 = v78->m128i_u64[1];
      v17 = v78 + 1;
      v18 = v86;
      while ( 1 )
      {
        if ( v16 <= v85 - v18 )
        {
          if ( v16 )
          {
            v66 = v14;
            memcpy(v18, v15, v16);
            v86 += v16;
            v14 = v66;
          }
          if ( v14 == v17 )
          {
LABEL_25:
            v14 = v78;
            break;
          }
        }
        else
        {
          v65 = v14;
          sub_CB6200((__int64)v84, (unsigned __int8 *)v15, v16);
          v14 = v65;
          if ( v65 == v17 )
            goto LABEL_25;
        }
        v15 = (void *)v17->m128i_i64[0];
        v16 = v17->m128i_u64[1];
        if ( v85 == v86 )
        {
          v59 = v14;
          v67 = (void *)v17->m128i_i64[0];
          sub_CB6200((__int64)v84, (unsigned __int8 *)" ", 1u);
          v18 = v86;
          v15 = v67;
          v14 = v59;
        }
        else
        {
          *v86 = 32;
          v18 = ++v86;
        }
        ++v17;
      }
    }
    if ( v14 )
      j_j___libc_free_0((unsigned __int64)v14);
    a1->m128i_i64[0] = (__int64)a1[1].m128i_i64;
    v19 = (unsigned __int64)v81;
    if ( v81 == &v83 )
      goto LABEL_78;
    goto LABEL_29;
  }
  v9 = sub_2E890A0(a3, a5, 0);
  if ( v9 >= 0 && v9 == a5 )
  {
    v10 = *(_QWORD *)(a4 + 24);
    v11 = v10 & 7;
    switch ( (char)v11 )
    {
      case 0:
        BUG();
      case 1:
        v21 = 6;
        v22 = "reguse";
        break;
      case 2:
        v21 = 6;
        v22 = "regdef";
        break;
      case 3:
        v21 = 9;
        v22 = "regdef-ec";
        break;
      case 4:
        v21 = 7;
        v22 = "clobber";
        break;
      case 5:
        v21 = 3;
        v22 = "imm";
        break;
      case 6:
      case 7:
        v21 = 3;
        v22 = "mem";
        break;
    }
    v23 = v86;
    if ( v21 > v85 - v86 )
    {
      v70 = v10;
      sub_CB6200((__int64)v84, (unsigned __int8 *)v22, v21);
      LODWORD(v10) = v70;
    }
    else
    {
      if ( (unsigned int)v21 >= 8 )
      {
        v41 = (unsigned __int64)(v86 + 8) & 0xFFFFFFFFFFFFFFF8LL;
        *(_QWORD *)v86 = *(_QWORD *)v22;
        *(_QWORD *)&v23[v21 - 8] = *(_QWORD *)&v22[v21 - 8];
        v42 = &v23[-v41];
        v43 = (char *)(v22 - v42);
        if ( (((_DWORD)v21 + (_DWORD)v42) & 0xFFFFFFF8) >= 8 )
        {
          v44 = (v21 + (_DWORD)v42) & 0xFFFFFFF8;
          v45 = 0;
          do
          {
            v46 = v45;
            v45 += 8;
            *(_QWORD *)(v41 + v46) = *(_QWORD *)&v43[v46];
          }
          while ( v45 < v44 );
        }
      }
      else if ( (v21 & 4) != 0 )
      {
        *(_DWORD *)v86 = *(_DWORD *)v22;
        *(_DWORD *)&v23[(unsigned int)v21 - 4] = *(_DWORD *)&v22[(unsigned int)v21 - 4];
      }
      else
      {
        *v86 = *v22;
        *(_WORD *)&v23[(unsigned int)v21 - 2] = *(_WORD *)&v22[(unsigned int)v21 - 2];
      }
      v86 += v21;
    }
    v24 = (unsigned int)v10 >> 31;
    if ( v11 != 5 )
    {
      if ( v11 != 6 )
      {
        if ( (int)v10 >= 0 )
        {
          if ( (v10 & 0x3FFF0000) != 0 )
          {
            v25 = (WORD1(v10) & 0x3FFFu) - 1;
            v26 = v86;
            if ( a6 )
            {
              if ( v86 >= v85 )
              {
                v51 = v10;
                v75 = (WORD1(v10) & 0x3FFF) - 1;
                v49 = sub_CB5D20((__int64)v84, 58);
                v25 = v75;
                LODWORD(v10) = v51;
                v27 = (_QWORD *)v49;
              }
              else
              {
                v27 = v84;
                ++v86;
                *v26 = 58;
              }
              v68 = (__int64)v27;
              v28 = (char *)(*(_QWORD *)(a6 + 80)
                           + *(unsigned int *)(**(_QWORD **)(*(_QWORD *)(a6 + 280) + 8 * v25) + 16LL));
              if ( v28 )
              {
                v52 = v10;
                v29 = strlen(v28);
                LODWORD(v10) = v52;
                v30 = *(void **)(v68 + 32);
                if ( v29 > *(_QWORD *)(v68 + 24) - (_QWORD)v30 )
                {
                  sub_CB6200(v68, (unsigned __int8 *)v28, v29);
                  LODWORD(v10) = v52;
                }
                else if ( v29 )
                {
                  v60 = v29;
                  memcpy(v30, v28, v29);
                  *(_QWORD *)(v68 + 32) += v60;
                  LODWORD(v10) = v52;
                }
              }
            }
            else
            {
              if ( (unsigned __int64)(v85 - v86) <= 2 )
              {
                v54 = v10;
                v76 = (WORD1(v10) & 0x3FFF) - 1;
                v50 = sub_CB6200((__int64)v84, ":RC", 3u);
                LODWORD(v25) = v76;
                LODWORD(v10) = v54;
                v48 = (_QWORD *)v50;
              }
              else
              {
                v86[2] = 67;
                v48 = v84;
                *(_WORD *)v26 = 21050;
                v86 += 3;
              }
              v57 = v10;
              sub_CB59D0((__int64)v48, (unsigned int)v25);
              LODWORD(v10) = v57;
            }
          }
          goto LABEL_74;
        }
LABEL_86:
        v31 = v86;
        v32 = WORD1(v10) & 0x7FFF;
        if ( (unsigned __int64)(v85 - v86) <= 8 )
        {
          v71 = v10;
          v35 = sub_CB6200((__int64)v84, " tiedto:$", 9u);
          LODWORD(v10) = v71;
          v33 = (_QWORD *)v35;
        }
        else
        {
          v86[8] = 36;
          v33 = v84;
          *(_QWORD *)v31 = 0x3A6F746465697420LL;
          v86 += 9;
        }
        v69 = v10;
        sub_CB59D0((__int64)v33, v32);
        LODWORD(v10) = v69;
LABEL_74:
        if ( ((v11 & 0xFD) == 1 || v11 == 2) && (v10 & 0x40000000) != 0 )
        {
          v34 = v86;
          if ( (unsigned __int64)(v85 - v86) <= 8 )
          {
            sub_CB6200((__int64)v84, " foldable", 9u);
          }
          else
          {
            v86[8] = 101;
            *(_QWORD *)v34 = 0x6C6261646C6F6620LL;
            v86 += 9;
          }
        }
LABEL_77:
        a1->m128i_i64[0] = (__int64)a1[1].m128i_i64;
        v19 = (unsigned __int64)v81;
        if ( v81 == &v83 )
        {
LABEL_78:
          a1[1] = _mm_load_si128(&v83);
          goto LABEL_30;
        }
LABEL_29:
        a1->m128i_i64[0] = v19;
        a1[1].m128i_i64[0] = v83.m128i_i64[0];
LABEL_30:
        v20 = v82;
        v81 = &v83;
        v82 = 0;
        a1->m128i_i64[1] = v20;
        v83.m128i_i8[0] = 0;
        goto LABEL_32;
      }
      v36 = WORD1(v10) & 0x7FFF;
      if ( v85 == v86 )
      {
        v63 = v10;
        v74 = (unsigned int)v10 >> 31;
        v47 = sub_CB6200((__int64)v84, (unsigned __int8 *)":", 1u);
        v24 = v74;
        LODWORD(v10) = v63;
        v37 = (_QWORD *)v47;
      }
      else
      {
        *v86 = 58;
        v37 = v84;
        ++v86;
      }
      v55 = v10;
      v61 = v24;
      v72 = (__int64)v37;
      v38 = (unsigned __int8 *)sub_2E862C0(v36);
      v24 = v61;
      LODWORD(v10) = v55;
      v40 = *(void **)(v72 + 32);
      if ( *(_QWORD *)(v72 + 24) - (_QWORD)v40 < v39 )
      {
        sub_CB6200(v72, v38, v39);
        v24 = v61;
        LODWORD(v10) = v55;
      }
      else if ( v39 )
      {
        v53 = v55;
        v56 = v61;
        v62 = v72;
        v73 = v39;
        memcpy(v40, v38, v39);
        LODWORD(v10) = v53;
        v24 = v56;
        *(_QWORD *)(v62 + 32) += v73;
      }
    }
    if ( !v24 )
      goto LABEL_77;
    goto LABEL_86;
  }
  a1->m128i_i64[0] = (__int64)a1[1].m128i_i64;
  sub_2FDCEA0(a1->m128i_i64, byte_3F871B3, (__int64)byte_3F871B3);
LABEL_32:
  v84[0] = &unk_49DD210;
  sub_CB5840((__int64)v84);
  if ( v81 != &v83 )
    j_j___libc_free_0((unsigned __int64)v81);
  return a1;
}
