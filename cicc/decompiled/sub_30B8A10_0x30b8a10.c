// Function: sub_30B8A10
// Address: 0x30b8a10
//
__int64 __fastcall sub_30B8A10(__int64 a1, __int64 *a2, __int64 a3, __int64 a4)
{
  __int64 v6; // rax
  __int64 v7; // r15
  __m128i *v8; // rdx
  __m128i si128; // xmm0
  __int64 v10; // r12
  const char *v11; // rax
  size_t v12; // rdx
  _WORD *v13; // rdi
  unsigned __int8 *v14; // rsi
  unsigned __int64 v15; // rax
  __int64 v16; // r13
  __int64 v17; // r12
  __int64 v18; // rbx
  __int64 v20; // r14
  __int64 v21; // r13
  __int64 j; // r15
  __int64 v23; // rbx
  unsigned __int8 v24; // al
  __int64 v25; // rdi
  __int64 v26; // r8
  int v27; // esi
  int v28; // esi
  unsigned int v29; // ecx
  __int64 *v30; // rdx
  __int64 v31; // r9
  __int64 *v32; // rbx
  char *v33; // r14
  __int64 v34; // rsi
  __int64 *v35; // r15
  __int64 v36; // rax
  __int64 v37; // r12
  _QWORD *v38; // r15
  _BYTE *v39; // rax
  __int64 v40; // rdx
  __int64 v41; // rax
  _BYTE *v42; // r8
  _BYTE *v43; // rax
  __m128i *v44; // rdx
  __m128i v45; // xmm0
  __int64 v46; // r8
  const char *v47; // rax
  size_t v48; // rdx
  __int64 v49; // r8
  unsigned __int8 *v50; // rsi
  _BYTE *v51; // rax
  _BYTE *v52; // rdi
  __m128i *v53; // rdx
  __int64 v54; // r8
  _BYTE *v55; // rax
  _QWORD *v56; // rax
  __int64 v57; // r9
  __m128i *v58; // rdx
  __m128i v59; // xmm0
  __int64 v60; // rax
  unsigned __int64 v61; // rax
  __int64 v62; // rax
  void *v63; // rdx
  __int64 v64; // r15
  _BYTE *v65; // rax
  __m128i *v66; // rdx
  __m128i v67; // xmm0
  __int64 v68; // rax
  __int64 v69; // rbx
  __int64 v70; // r13
  __int64 v71; // r15
  __int64 v72; // r12
  _BYTE *v73; // rax
  __m128i v74; // xmm0
  __int64 v75; // r12
  _QWORD *v76; // rdx
  _QWORD *v77; // rdx
  _BYTE *v78; // rax
  unsigned int v79; // edi
  __int64 v80; // rbx
  __int64 v81; // r13
  __int64 v82; // r12
  __int64 v83; // r15
  _BYTE *v84; // rax
  int i; // edx
  int v86; // r10d
  __int64 *v87; // [rsp+0h] [rbp-F0h]
  __int64 v88; // [rsp+8h] [rbp-E8h]
  __int64 v89; // [rsp+10h] [rbp-E0h]
  __int64 v91; // [rsp+20h] [rbp-D0h]
  __int64 v92; // [rsp+28h] [rbp-C8h]
  int v93; // [rsp+30h] [rbp-C0h]
  int v94; // [rsp+38h] [rbp-B8h]
  __int64 *v95; // [rsp+38h] [rbp-B8h]
  __int64 v96; // [rsp+40h] [rbp-B0h]
  __int64 v97; // [rsp+48h] [rbp-A8h]
  __int64 v98; // [rsp+48h] [rbp-A8h]
  __int64 v99; // [rsp+48h] [rbp-A8h]
  size_t v100; // [rsp+48h] [rbp-A8h]
  __int64 v101; // [rsp+50h] [rbp-A0h]
  __int64 v102; // [rsp+58h] [rbp-98h]
  size_t v103; // [rsp+58h] [rbp-98h]
  _BYTE *v104; // [rsp+60h] [rbp-90h] BYREF
  __int64 v105; // [rsp+68h] [rbp-88h]
  _BYTE v106[32]; // [rsp+70h] [rbp-80h] BYREF
  _BYTE *v107; // [rsp+90h] [rbp-60h] BYREF
  __int64 v108; // [rsp+98h] [rbp-58h]
  _BYTE v109[80]; // [rsp+A0h] [rbp-50h] BYREF

  v88 = sub_BC1CD0(a4, &unk_4F881D0, a3) + 8;
  v6 = sub_BC1CD0(a4, &unk_4F875F0, a3);
  v7 = *a2;
  v89 = v6;
  v8 = *(__m128i **)(*a2 + 32);
  if ( *(_QWORD *)(*a2 + 24) - (_QWORD)v8 <= 0x1Bu )
  {
    v10 = sub_CB6200(v7, "Delinearization on function ", 0x1Cu);
  }
  else
  {
    si128 = _mm_load_si128((const __m128i *)&xmmword_4289610);
    v10 = *a2;
    qmemcpy(&v8[1], "on function ", 12);
    *v8 = si128;
    *(_QWORD *)(v7 + 32) += 28LL;
  }
  v11 = sub_BD5D20(a3);
  v13 = *(_WORD **)(v10 + 32);
  v14 = (unsigned __int8 *)v11;
  v15 = *(_QWORD *)(v10 + 24) - (_QWORD)v13;
  if ( v15 < v12 )
  {
    v62 = sub_CB6200(v10, v14, v12);
    v13 = *(_WORD **)(v62 + 32);
    v10 = v62;
    v15 = *(_QWORD *)(v62 + 24) - (_QWORD)v13;
  }
  else if ( v12 )
  {
    v103 = v12;
    memcpy(v13, v14, v12);
    v13 = (_WORD *)(v103 + *(_QWORD *)(v10 + 32));
    v61 = *(_QWORD *)(v10 + 24) - (_QWORD)v13;
    *(_QWORD *)(v10 + 32) = v13;
    if ( v61 > 1 )
      goto LABEL_6;
    goto LABEL_77;
  }
  if ( v15 > 1 )
  {
LABEL_6:
    *v13 = 2618;
    *(_QWORD *)(v10 + 32) += 2LL;
    goto LABEL_7;
  }
LABEL_77:
  sub_CB6200(v10, (unsigned __int8 *)":\n", 2u);
LABEL_7:
  v16 = *(_QWORD *)(a3 + 80);
  v17 = a3 + 72;
  if ( a3 + 72 == v16 )
  {
    v18 = 0;
  }
  else
  {
    if ( !v16 )
      BUG();
    while ( 1 )
    {
      v18 = *(_QWORD *)(v16 + 32);
      if ( v18 != v16 + 24 )
        break;
      v16 = *(_QWORD *)(v16 + 8);
      if ( v17 == v16 )
        goto LABEL_13;
      if ( !v16 )
        BUG();
    }
  }
  if ( v17 != v16 )
  {
    v20 = v16;
    v21 = v7;
    j = v18;
    v23 = v17;
    while ( 1 )
    {
      if ( !j )
        BUG();
      v24 = *(_BYTE *)(j - 24);
      if ( (unsigned __int8)(v24 - 61) > 2u )
        goto LABEL_57;
      v25 = *(_QWORD *)(j + 16);
      v26 = *(_QWORD *)(v89 + 16);
      v27 = *(_DWORD *)(v89 + 32);
      if ( !v27 )
        goto LABEL_57;
      v28 = v27 - 1;
      v29 = v28 & (((unsigned int)v25 >> 9) ^ ((unsigned int)v25 >> 4));
      v30 = (__int64 *)(v26 + 16LL * v29);
      v31 = *v30;
      if ( v25 != *v30 )
      {
        for ( i = 1; ; i = v86 )
        {
          if ( v31 == -4096 )
            goto LABEL_57;
          v86 = i + 1;
          v29 = v28 & (i + v29);
          v30 = (__int64 *)(v26 + 16LL * v29);
          v31 = *v30;
          if ( v25 == *v30 )
            break;
        }
      }
      if ( v30[1] )
        break;
LABEL_57:
      for ( j = *(_QWORD *)(j + 8); ; j = *(_QWORD *)(v20 + 32) )
      {
        v60 = v20 - 24;
        if ( !v20 )
          v60 = 0;
        if ( j != v60 + 48 )
          break;
        v20 = *(_QWORD *)(v20 + 8);
        if ( v23 == v20 )
          goto LABEL_13;
        if ( !v20 )
          BUG();
      }
      if ( v23 == v20 )
        goto LABEL_13;
    }
    v102 = j;
    v101 = j - 24;
    v92 = v23;
    v32 = (__int64 *)v88;
    v91 = v20;
    v33 = (char *)v30[1];
    while ( 1 )
    {
      v34 = 0;
      if ( v24 > 0x1Cu )
      {
        if ( v24 == 61 || v24 == 62 )
        {
          v34 = *(_QWORD *)(v102 - 56);
        }
        else if ( v24 == 63 )
        {
          v34 = *(_QWORD *)(v101 - 32LL * (*(_DWORD *)(v102 - 20) & 0x7FFFFFF));
        }
      }
      v35 = sub_DDFBA0((__int64)v32, v34, v33);
      v36 = sub_D97190((__int64)v32, (__int64)v35);
      v37 = v36;
      if ( *(_WORD *)(v36 + 24) != 15 )
      {
LABEL_56:
        j = v102;
        v23 = v92;
        v20 = v91;
        goto LABEL_57;
      }
      v38 = sub_DCC810(v32, (__int64)v35, v36, 0, 0);
      v39 = *(_BYTE **)(v21 + 32);
      if ( *(_BYTE **)(v21 + 24) == v39 )
      {
        sub_CB6200(v21, (unsigned __int8 *)"\n", 1u);
        v40 = *(_QWORD *)(v21 + 32);
        if ( (unsigned __int64)(*(_QWORD *)(v21 + 24) - v40) > 4 )
        {
LABEL_31:
          *(_DWORD *)v40 = 1953721929;
          v42 = (_BYTE *)v21;
          *(_BYTE *)(v40 + 4) = 58;
          *(_QWORD *)(v21 + 32) += 5LL;
          goto LABEL_32;
        }
      }
      else
      {
        *v39 = 10;
        v40 = *(_QWORD *)(v21 + 32) + 1LL;
        v41 = *(_QWORD *)(v21 + 24);
        *(_QWORD *)(v21 + 32) = v40;
        if ( (unsigned __int64)(v41 - v40) > 4 )
          goto LABEL_31;
      }
      v42 = (_BYTE *)sub_CB6200(v21, "Inst:", 5u);
LABEL_32:
      v97 = (__int64)v42;
      sub_A69870(v101, v42, 0);
      v43 = *(_BYTE **)(v97 + 32);
      if ( *(_BYTE **)(v97 + 24) == v43 )
      {
        sub_CB6200(v97, (unsigned __int8 *)"\n", 1u);
      }
      else
      {
        *v43 = 10;
        ++*(_QWORD *)(v97 + 32);
      }
      v44 = *(__m128i **)(v21 + 32);
      if ( *(_QWORD *)(v21 + 24) - (_QWORD)v44 <= 0x14u )
      {
        v46 = sub_CB6200(v21, "In Loop with Header: ", 0x15u);
      }
      else
      {
        v45 = _mm_load_si128((const __m128i *)&xmmword_4289620);
        v44[1].m128i_i32[0] = 980575588;
        v46 = v21;
        v44[1].m128i_i8[4] = 32;
        *v44 = v45;
        *(_QWORD *)(v21 + 32) += 21LL;
      }
      v98 = v46;
      v47 = sub_BD5D20(**((_QWORD **)v33 + 4));
      v49 = v98;
      v50 = (unsigned __int8 *)v47;
      v51 = *(_BYTE **)(v98 + 24);
      v52 = *(_BYTE **)(v98 + 32);
      if ( v51 - v52 < v48 )
      {
        v49 = sub_CB6200(v98, v50, v48);
        v51 = *(_BYTE **)(v49 + 24);
        v52 = *(_BYTE **)(v49 + 32);
      }
      else if ( v48 )
      {
        v96 = v98;
        v100 = v48;
        memcpy(v52, v50, v48);
        v49 = v96;
        v52 = (_BYTE *)(*(_QWORD *)(v96 + 32) + v100);
        v51 = *(_BYTE **)(v96 + 24);
        *(_QWORD *)(v96 + 32) = v52;
      }
      if ( v52 == v51 )
      {
        sub_CB6200(v49, (unsigned __int8 *)"\n", 1u);
      }
      else
      {
        *v52 = 10;
        ++*(_QWORD *)(v49 + 32);
      }
      v53 = *(__m128i **)(v21 + 32);
      if ( *(_QWORD *)(v21 + 24) - (_QWORD)v53 <= 0xFu )
      {
        v54 = sub_CB6200(v21, "AccessFunction: ", 0x10u);
      }
      else
      {
        v54 = v21;
        *v53 = _mm_load_si128((const __m128i *)&xmmword_4289630);
        *(_QWORD *)(v21 + 32) += 16LL;
      }
      v99 = v54;
      sub_D955C0((__int64)v38, v54);
      v55 = *(_BYTE **)(v99 + 32);
      if ( *(_BYTE **)(v99 + 24) == v55 )
      {
        sub_CB6200(v99, (unsigned __int8 *)"\n", 1u);
      }
      else
      {
        *v55 = 10;
        ++*(_QWORD *)(v99 + 32);
      }
      v107 = v109;
      v104 = v106;
      v105 = 0x300000000LL;
      v108 = 0x300000000LL;
      v56 = sub_DCADF0(v32, v101);
      sub_30B8650(v32, (__int64)v38, (__int64)&v104, (__int64)&v107, (__int64)v56, v57);
      if ( (_DWORD)v105 && (unsigned int)v105 == (unsigned __int64)(unsigned int)v108 && (_DWORD)v108 )
      {
        v63 = *(void **)(v21 + 32);
        if ( *(_QWORD *)(v21 + 24) - (_QWORD)v63 <= 0xCu )
        {
          v64 = sub_CB6200(v21, "Base offset: ", 0xDu);
        }
        else
        {
          v64 = v21;
          qmemcpy(v63, "Base offset: ", 13);
          *(_QWORD *)(v21 + 32) += 13LL;
        }
        sub_D955C0(v37, v64);
        v65 = *(_BYTE **)(v64 + 32);
        if ( *(_BYTE **)(v64 + 24) == v65 )
        {
          sub_CB6200(v64, (unsigned __int8 *)"\n", 1u);
        }
        else
        {
          *v65 = 10;
          ++*(_QWORD *)(v64 + 32);
        }
        v66 = *(__m128i **)(v21 + 32);
        if ( *(_QWORD *)(v21 + 24) - (_QWORD)v66 <= 0x15u )
        {
          sub_CB6200(v21, "ArrayDecl[UnknownSize]", 0x16u);
          v68 = *(_QWORD *)(v21 + 32);
        }
        else
        {
          v67 = _mm_load_si128((const __m128i *)&xmmword_4289640);
          v66[1].m128i_i32[0] = 2053722990;
          v66[1].m128i_i16[2] = 23909;
          *v66 = v67;
          v68 = *(_QWORD *)(v21 + 32) + 22LL;
          *(_QWORD *)(v21 + 32) = v68;
        }
        v94 = v105;
        v93 = v105 - 1;
        if ( (int)v105 - 1 > 0 )
        {
          v87 = v32;
          v69 = v21;
          v70 = 0;
          v71 = 8LL * (unsigned int)(v105 - 2) + 8;
          do
          {
            while ( 1 )
            {
              if ( *(_QWORD *)(v69 + 24) == v68 )
              {
                v72 = sub_CB6200(v69, (unsigned __int8 *)"[", 1u);
              }
              else
              {
                *(_BYTE *)v68 = 91;
                v72 = v69;
                ++*(_QWORD *)(v69 + 32);
              }
              sub_D955C0(*(_QWORD *)&v107[v70], v72);
              v73 = *(_BYTE **)(v72 + 32);
              if ( *(_BYTE **)(v72 + 24) == v73 )
                break;
              v70 += 8;
              *v73 = 93;
              ++*(_QWORD *)(v72 + 32);
              v68 = *(_QWORD *)(v69 + 32);
              if ( v71 == v70 )
                goto LABEL_93;
            }
            v70 += 8;
            sub_CB6200(v72, (unsigned __int8 *)"]", 1u);
            v68 = *(_QWORD *)(v69 + 32);
          }
          while ( v71 != v70 );
LABEL_93:
          v21 = v69;
          v32 = v87;
        }
        if ( (unsigned __int64)(*(_QWORD *)(v21 + 24) - v68) <= 0x11 )
        {
          v75 = sub_CB6200(v21, " with elements of ", 0x12u);
        }
        else
        {
          v74 = _mm_load_si128((const __m128i *)&xmmword_4289660);
          v75 = v21;
          *(_WORD *)(v68 + 16) = 8294;
          *(__m128i *)v68 = v74;
          *(_QWORD *)(v21 + 32) += 18LL;
        }
        sub_D955C0(*(_QWORD *)&v107[8 * v93], v75);
        v76 = *(_QWORD **)(v75 + 32);
        if ( *(_QWORD *)(v75 + 24) - (_QWORD)v76 <= 7u )
        {
          sub_CB6200(v75, " bytes.\n", 8u);
        }
        else
        {
          *v76 = 0xA2E736574796220LL;
          *(_QWORD *)(v75 + 32) += 8LL;
        }
        v77 = *(_QWORD **)(v21 + 32);
        if ( *(_QWORD *)(v21 + 24) - (_QWORD)v77 <= 7u )
        {
          sub_CB6200(v21, "ArrayRef", 8u);
          v78 = *(_BYTE **)(v21 + 32);
        }
        else
        {
          *v77 = 0x6665527961727241LL;
          v78 = (_BYTE *)(*(_QWORD *)(v21 + 32) + 8LL);
          *(_QWORD *)(v21 + 32) = v78;
        }
        v79 = v94;
        if ( v94 > 0 )
        {
          v95 = v32;
          v80 = v21;
          v81 = 0;
          v82 = 8LL * v79;
          do
          {
            while ( 1 )
            {
              if ( *(_BYTE **)(v80 + 24) == v78 )
              {
                v83 = sub_CB6200(v80, (unsigned __int8 *)"[", 1u);
              }
              else
              {
                *v78 = 91;
                v83 = v80;
                ++*(_QWORD *)(v80 + 32);
              }
              sub_D955C0(*(_QWORD *)&v104[v81], v83);
              v84 = *(_BYTE **)(v83 + 32);
              if ( *(_BYTE **)(v83 + 24) == v84 )
                break;
              v81 += 8;
              *v84 = 93;
              ++*(_QWORD *)(v83 + 32);
              v78 = *(_BYTE **)(v80 + 32);
              if ( v82 == v81 )
                goto LABEL_107;
            }
            v81 += 8;
            sub_CB6200(v83, (unsigned __int8 *)"]", 1u);
            v78 = *(_BYTE **)(v80 + 32);
          }
          while ( v82 != v81 );
LABEL_107:
          v21 = v80;
          v32 = v95;
        }
        if ( v78 == *(_BYTE **)(v21 + 24) )
        {
          sub_CB6200(v21, (unsigned __int8 *)"\n", 1u);
        }
        else
        {
          *v78 = 10;
          ++*(_QWORD *)(v21 + 32);
        }
      }
      else
      {
        v58 = *(__m128i **)(v21 + 32);
        if ( *(_QWORD *)(v21 + 24) - (_QWORD)v58 <= 0x15u )
        {
          sub_CB6200(v21, "failed to delinearize\n", 0x16u);
        }
        else
        {
          v59 = _mm_load_si128((const __m128i *)&xmmword_4289650);
          v58[1].m128i_i32[0] = 2053730913;
          v58[1].m128i_i16[2] = 2661;
          *v58 = v59;
          *(_QWORD *)(v21 + 32) += 22LL;
        }
      }
      if ( v107 != v109 )
        _libc_free((unsigned __int64)v107);
      if ( v104 != v106 )
        _libc_free((unsigned __int64)v104);
      v33 = *(char **)v33;
      if ( !v33 )
        goto LABEL_56;
      v24 = *(_BYTE *)(v102 - 24);
    }
  }
LABEL_13:
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
