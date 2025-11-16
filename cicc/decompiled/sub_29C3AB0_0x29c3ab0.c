// Function: sub_29C3AB0
// Address: 0x29c3ab0
//
__int64 __fastcall sub_29C3AB0(
        __int64 a1,
        __int64 a2,
        __int64 a3,
        unsigned __int8 *a4,
        size_t a5,
        char a6,
        unsigned __int8 *a7,
        size_t a8,
        unsigned __int64 *a9)
{
  unsigned __int8 **v10; // rdx
  unsigned __int8 **v11; // r15
  __int64 v12; // rdx
  unsigned __int8 *v13; // r13
  __int64 v14; // rsi
  unsigned int v15; // ecx
  __int64 v16; // rax
  unsigned __int8 *v17; // r9
  __int64 v18; // rax
  __int64 v19; // rax
  const char *v20; // rax
  __int64 v21; // rdi
  size_t v22; // rdx
  size_t v23; // r12
  char *v24; // r14
  __int64 v25; // rsi
  __int64 v26; // rax
  unsigned int v27; // ecx
  __int64 v28; // rdx
  unsigned __int8 *v29; // r9
  __int64 v30; // rax
  __int64 v31; // r14
  __int64 v32; // rdx
  __m128i *v33; // rdi
  unsigned __int64 v34; // rax
  __m128i si128; // xmm0
  __int64 v36; // rdx
  _WORD *v37; // rdi
  unsigned __int64 v38; // rax
  _QWORD *v39; // rdi
  unsigned __int64 v40; // rax
  _WORD *v41; // rdi
  unsigned __int64 v42; // rax
  size_t v44; // rdx
  int v45; // edx
  void *v46; // rdi
  __int64 v47; // rax
  void *v48; // rdi
  __int64 v49; // r14
  _BYTE *v50; // r14
  __int64 v51; // rax
  void *v52; // rdi
  __int64 v53; // r14
  __int64 v54; // rax
  void *v55; // rdi
  __int64 v56; // r13
  __int64 v57; // rax
  void *v58; // rdi
  __int64 v59; // r12
  _QWORD *v60; // rax
  __int64 v61; // rax
  __int64 v62; // rax
  __int64 v63; // rax
  __int64 v64; // rax
  __int64 v65; // rax
  __int64 v66; // rax
  __int64 v67; // rax
  __int64 v68; // rax
  int v69; // eax
  int v70; // r8d
  _QWORD *v71; // rax
  _QWORD *v72; // rax
  size_t v73; // r12
  __int64 v74; // rdi
  unsigned __int16 *v75; // r12
  __m128i *v76; // r13
  __int64 v77; // rax
  __int64 v78; // rax
  __int64 v79; // rax
  __int64 v80; // rax
  _QWORD *v81; // rax
  size_t v82; // r12
  __int64 v83; // rdi
  unsigned __int16 *v84; // r12
  __m128i *v85; // r13
  _QWORD *v86; // rdi
  _QWORD *v87; // rdi
  int v88; // r8d
  _QWORD *v89; // rdi
  _QWORD *v90; // rdi
  unsigned __int64 v93; // [rsp+30h] [rbp-250h]
  unsigned __int8 v94; // [rsp+38h] [rbp-248h]
  unsigned __int8 *v95; // [rsp+48h] [rbp-238h]
  char *src; // [rsp+58h] [rbp-228h]
  __int64 v99; // [rsp+68h] [rbp-218h]
  unsigned __int8 **v100; // [rsp+70h] [rbp-210h]
  size_t n; // [rsp+78h] [rbp-208h]
  _QWORD *v102; // [rsp+80h] [rbp-200h] BYREF
  unsigned __int64 v103; // [rsp+88h] [rbp-1F8h]
  _QWORD v104[2]; // [rsp+90h] [rbp-1F0h] BYREF
  _QWORD *v105; // [rsp+A0h] [rbp-1E0h] BYREF
  size_t v106; // [rsp+A8h] [rbp-1D8h]
  _QWORD v107[2]; // [rsp+B0h] [rbp-1D0h] BYREF
  __int64 *v108; // [rsp+C0h] [rbp-1C0h] BYREF
  __int64 v109; // [rsp+C8h] [rbp-1B8h]
  __int64 v110; // [rsp+D0h] [rbp-1B0h] BYREF
  unsigned int v111; // [rsp+D8h] [rbp-1A8h]
  size_t v112; // [rsp+E0h] [rbp-1A0h] BYREF
  __int64 v113; // [rsp+E8h] [rbp-198h]
  __int64 v114; // [rsp+F0h] [rbp-190h]
  __int64 v115; // [rsp+F8h] [rbp-188h]
  unsigned int v116; // [rsp+100h] [rbp-180h]
  __m128i *v117[3]; // [rsp+110h] [rbp-170h] BYREF
  unsigned __int16 v118; // [rsp+128h] [rbp-158h] BYREF
  const char *v119; // [rsp+130h] [rbp-150h]
  __int64 v120; // [rsp+138h] [rbp-148h]
  __m128i *v121[3]; // [rsp+150h] [rbp-130h] BYREF
  _BYTE v122[40]; // [rsp+168h] [rbp-118h] BYREF
  __m128i *v123[3]; // [rsp+190h] [rbp-F0h] BYREF
  _BYTE v124[40]; // [rsp+1A8h] [rbp-D8h] BYREF
  __m128i *v125[3]; // [rsp+1D0h] [rbp-B0h] BYREF
  unsigned __int16 v126; // [rsp+1E8h] [rbp-98h] BYREF
  char *v127; // [rsp+1F0h] [rbp-90h]
  size_t v128; // [rsp+1F8h] [rbp-88h]
  __m128i *v129[3]; // [rsp+210h] [rbp-70h] BYREF
  unsigned __int16 v130; // [rsp+228h] [rbp-58h] BYREF
  char *v131; // [rsp+230h] [rbp-50h]
  __int64 v132; // [rsp+238h] [rbp-48h]
  _BYTE v133[48]; // [rsp+250h] [rbp-30h] BYREF

  v10 = *(unsigned __int8 ***)(a2 + 32);
  v100 = &v10[2 * *(unsigned int *)(a2 + 40)];
  if ( v10 != v100 )
  {
    v94 = 1;
    v11 = *(unsigned __int8 ***)(a2 + 32);
    while ( 1 )
    {
      if ( !*((_BYTE *)v11 + 8) )
      {
        v12 = *(unsigned int *)(a3 + 24);
        v13 = *v11;
        v14 = *(_QWORD *)(a3 + 8);
        if ( (_DWORD)v12 )
        {
          v15 = (v12 - 1) & (((unsigned int)v13 >> 9) ^ ((unsigned int)v13 >> 4));
          v16 = v14 + 16LL * v15;
          v17 = *(unsigned __int8 **)v16;
          if ( v13 == *(unsigned __int8 **)v16 )
          {
LABEL_6:
            if ( v16 != v14 + 16 * v12 )
            {
              v18 = *(_QWORD *)(a3 + 32) + 32LL * *(unsigned int *)(v16 + 8);
              if ( v18 != *(_QWORD *)(a3 + 32) + 32LL * *(unsigned int *)(a3 + 40) && !*(_QWORD *)(v18 + 24) )
                goto LABEL_44;
            }
          }
          else
          {
            v69 = 1;
            while ( v17 != (unsigned __int8 *)-4096LL )
            {
              v70 = v69 + 1;
              v15 = (v12 - 1) & (v69 + v15);
              v16 = v14 + 16LL * v15;
              v17 = *(unsigned __int8 **)v16;
              if ( v13 == *(unsigned __int8 **)v16 )
                goto LABEL_6;
              v69 = v70;
            }
          }
        }
        v19 = sub_B43CB0((__int64)*v11);
        v20 = sub_BD5D20(v19);
        v21 = *((_QWORD *)v13 + 5);
        n = 7;
        v95 = (unsigned __int8 *)v20;
        v23 = v22;
        v93 = v22;
        src = "no-name";
        if ( (*(_BYTE *)(v21 + 7) & 0x10) != 0 )
        {
          src = (char *)sub_BD5D20(v21);
          n = v44;
        }
        v24 = sub_B458E0((unsigned int)*v13 - 29);
        v25 = *(_QWORD *)(a1 + 8);
        v26 = *(unsigned int *)(a1 + 24);
        if ( !(_DWORD)v26 )
          goto LABEL_48;
        v27 = (v26 - 1) & (((unsigned int)v13 >> 9) ^ ((unsigned int)v13 >> 4));
        v28 = v25 + 16LL * v27;
        v29 = *(unsigned __int8 **)v28;
        if ( v13 != *(unsigned __int8 **)v28 )
        {
          v45 = 1;
          while ( v29 != (unsigned __int8 *)-4096LL )
          {
            v88 = v45 + 1;
            v27 = (v26 - 1) & (v45 + v27);
            v28 = v25 + 16LL * v27;
            v29 = *(unsigned __int8 **)v28;
            if ( v13 == *(unsigned __int8 **)v28 )
              goto LABEL_13;
            v45 = v88;
          }
LABEL_48:
          if ( !a6 )
          {
            if ( (_BYTE)qword_5008FC8 )
              v46 = sub_CB7330();
            else
              v46 = sub_CB72A0();
            v47 = sub_904010((__int64)v46, "WARNING: ");
            v48 = *(void **)(v47 + 32);
            v49 = v47;
            if ( a5 > *(_QWORD *)(v47 + 24) - (_QWORD)v48 )
            {
              v49 = sub_CB6200(v47, a4, a5);
            }
            else if ( a5 )
            {
              memcpy(v48, a4, a5);
              *(_QWORD *)(v49 + 32) += a5;
            }
            v50 = (_BYTE *)sub_904010(v49, " did not generate DILocation for ");
            sub_A69870((__int64)v13, v50, 0);
            v51 = sub_904010((__int64)v50, " (BB: ");
            v52 = *(void **)(v51 + 32);
            v53 = v51;
            if ( n > *(_QWORD *)(v51 + 24) - (_QWORD)v52 )
            {
              v53 = sub_CB6200(v51, (unsigned __int8 *)src, n);
            }
            else if ( n )
            {
              memcpy(v52, src, n);
              *(_QWORD *)(v53 + 32) += n;
            }
            v54 = sub_904010(v53, ", Fn: ");
            v55 = *(void **)(v54 + 32);
            v56 = v54;
            if ( v23 > *(_QWORD *)(v54 + 24) - (_QWORD)v55 )
            {
              v56 = sub_CB6200(v54, v95, v23);
            }
            else if ( v23 )
            {
              memcpy(v55, v95, v23);
              *(_QWORD *)(v56 + 32) += v23;
            }
            v57 = sub_904010(v56, ", File: ");
            v58 = *(void **)(v57 + 32);
            v59 = v57;
            if ( a8 > *(_QWORD *)(v57 + 24) - (_QWORD)v58 )
            {
              v59 = sub_CB6200(v57, a7, a8);
            }
            else if ( a8 )
            {
              memcpy(v58, a7, a8);
              *(_QWORD *)(v59 + 32) += a8;
            }
            sub_904010(v59, ")\n");
            goto LABEL_43;
          }
          sub_124B680(v117, "metadata", 8u);
          v118 = 5;
          v119 = "DILocation";
          v120 = 10;
          if ( !(unsigned __int8)sub_C6A630("DILocation", 10, 0) )
          {
            sub_C6B0E0((__int64 *)&v108, (__int64)"DILocation", 0xAu);
            sub_125BB70((__int64)&v112, (__int64)&v108);
            sub_C6BC50(&v118);
            sub_C6A4F0((__int64)&v118, (unsigned __int16 *)&v112);
            sub_C6BC50((unsigned __int16 *)&v112);
            if ( v108 != &v110 )
              j_j___libc_free_0((unsigned __int64)v108);
          }
          sub_124B680(v121, "fn-name", 7u);
          if ( v95 )
          {
            v112 = v23;
            v102 = v104;
            if ( v23 > 0xF )
            {
              v102 = (_QWORD *)sub_22409D0((__int64)&v102, &v112, 0);
              v90 = v102;
              v104[0] = v112;
            }
            else
            {
              if ( v23 == 1 )
              {
                LOBYTE(v104[0]) = *v95;
                v71 = v104;
                goto LABEL_91;
              }
              if ( !v23 )
              {
                v71 = v104;
                goto LABEL_91;
              }
              v90 = v104;
            }
            memcpy(v90, v95, v23);
            v93 = v112;
            v71 = v102;
LABEL_91:
            v103 = v93;
            *((_BYTE *)v71 + v93) = 0;
          }
          else
          {
            LOBYTE(v104[0]) = 0;
            v102 = v104;
            v103 = 0;
          }
          sub_125BB70((__int64)v122, (__int64)&v102);
          sub_124B680(v123, "bb-name", 7u);
          if ( src )
          {
            v105 = v107;
            v112 = n;
            if ( n > 0xF )
            {
              v105 = (_QWORD *)sub_22409D0((__int64)&v105, &v112, 0);
              v89 = v105;
              v107[0] = v112;
            }
            else
            {
              if ( n == 1 )
              {
                LOBYTE(v107[0]) = *src;
                v72 = v107;
                goto LABEL_96;
              }
              if ( !n )
              {
                v72 = v107;
                goto LABEL_96;
              }
              v89 = v107;
            }
            memcpy(v89, src, n);
            n = v112;
            v72 = v105;
LABEL_96:
            v106 = n;
            *((_BYTE *)v72 + n) = 0;
          }
          else
          {
            LOBYTE(v107[0]) = 0;
            v105 = v107;
            v106 = 0;
          }
          v73 = 0;
          sub_125BB70((__int64)v124, (__int64)&v105);
          sub_124B680(v125, "instr", 5u);
          if ( v24 )
            v73 = strlen(v24);
          v126 = 5;
          v127 = v24;
          v128 = v73;
          if ( !(unsigned __int8)sub_C6A630(v24, v73, 0) )
          {
            sub_C6B0E0((__int64 *)&v108, (__int64)v24, v73);
            sub_125BB70((__int64)&v112, (__int64)&v108);
            sub_C6BC50(&v126);
            sub_C6A4F0((__int64)&v126, (unsigned __int16 *)&v112);
            sub_C6BC50((unsigned __int16 *)&v112);
            if ( v108 != &v110 )
              j_j___libc_free_0((unsigned __int64)v108);
          }
          sub_124B680(v129, "action", 6u);
          v130 = 5;
          v131 = "not-generate";
          v132 = 12;
          if ( !(unsigned __int8)sub_C6A630("not-generate", 12, 0) )
          {
            sub_C6B0E0((__int64 *)&v108, (__int64)"not-generate", 0xCu);
            sub_125BB70((__int64)&v112, (__int64)&v108);
            sub_C6BC50(&v130);
            sub_C6A4F0((__int64)&v130, (unsigned __int16 *)&v112);
            sub_C6BC50((unsigned __int16 *)&v112);
            if ( v108 != &v110 )
              j_j___libc_free_0((unsigned __int64)v108);
          }
          sub_125BF70((__int64)&v108, (__int64)v117, 5);
          v108 = (__int64 *)((char *)v108 + 1);
          LOWORD(v112) = 7;
          v114 = v109;
          v113 = 1;
          v115 = v110;
          v109 = 0;
          v116 = v111;
          v110 = 0;
          v111 = 0;
          v74 = a9[1];
          if ( v74 == a9[2] )
          {
            sub_29C1F00(a9, (__int16 *)a9[1], (__int64)&v112);
          }
          else
          {
            if ( v74 )
            {
              sub_C6A4F0(v74, (unsigned __int16 *)&v112);
              v74 = a9[1];
            }
            a9[1] = v74 + 40;
          }
          v75 = (unsigned __int16 *)v133;
          sub_C6BC50((unsigned __int16 *)&v112);
          sub_C6B900((__int64)&v108);
          sub_C7D6A0(v109, (unsigned __int64)v111 << 6, 8);
          do
          {
            v75 -= 32;
            sub_C6BC50(v75 + 12);
            v76 = *(__m128i **)v75;
            if ( *(_QWORD *)v75 )
            {
              if ( (__m128i *)v76->m128i_i64[0] != &v76[1] )
                j_j___libc_free_0(v76->m128i_i64[0]);
              j_j___libc_free_0((unsigned __int64)v76);
            }
          }
          while ( v75 != (unsigned __int16 *)v117 );
LABEL_111:
          if ( v105 != v107 )
            j_j___libc_free_0((unsigned __int64)v105);
          if ( v102 != v104 )
            j_j___libc_free_0((unsigned __int64)v102);
LABEL_43:
          v94 = 0;
          goto LABEL_44;
        }
LABEL_13:
        if ( v28 == v25 + 16 * v26 )
          goto LABEL_48;
        v99 = *(_QWORD *)(a1 + 32);
        v30 = v99 + 16LL * *(unsigned int *)(v28 + 8);
        if ( v30 == v99 + 16LL * *(unsigned int *)(a1 + 40) )
          goto LABEL_48;
        if ( *(_BYTE *)(v30 + 8) )
          break;
      }
LABEL_44:
      v11 += 2;
      if ( v100 == v11 )
        return v94;
    }
    if ( !a6 )
    {
      if ( (_BYTE)qword_5008FC8 )
        v31 = (__int64)sub_CB7330();
      else
        v31 = (__int64)sub_CB72A0();
      v32 = *(_QWORD *)(v31 + 32);
      if ( (unsigned __int64)(*(_QWORD *)(v31 + 24) - v32) <= 8 )
      {
        v62 = sub_CB6200(v31, "WARNING: ", 9u);
        v33 = *(__m128i **)(v62 + 32);
        v31 = v62;
      }
      else
      {
        *(_BYTE *)(v32 + 8) = 32;
        *(_QWORD *)v32 = 0x3A474E494E524157LL;
        v33 = (__m128i *)(*(_QWORD *)(v31 + 32) + 9LL);
        *(_QWORD *)(v31 + 32) = v33;
      }
      v34 = *(_QWORD *)(v31 + 24) - (_QWORD)v33;
      if ( a5 > v34 )
      {
        v61 = sub_CB6200(v31, a4, a5);
        v33 = *(__m128i **)(v61 + 32);
        v31 = v61;
        v34 = *(_QWORD *)(v61 + 24) - (_QWORD)v33;
      }
      else if ( a5 )
      {
        memcpy(v33, a4, a5);
        v78 = *(_QWORD *)(v31 + 24);
        v33 = (__m128i *)(*(_QWORD *)(v31 + 32) + a5);
        *(_QWORD *)(v31 + 32) = v33;
        v34 = v78 - (_QWORD)v33;
      }
      if ( v34 <= 0x16 )
      {
        v31 = sub_CB6200(v31, " dropped DILocation of ", 0x17u);
      }
      else
      {
        si128 = _mm_load_si128((const __m128i *)&xmmword_439AB20);
        v33[1].m128i_i32[0] = 544108393;
        v33[1].m128i_i16[2] = 26223;
        v33[1].m128i_i8[6] = 32;
        *v33 = si128;
        *(_QWORD *)(v31 + 32) += 23LL;
      }
      sub_A69870((__int64)v13, (_BYTE *)v31, 0);
      v36 = *(_QWORD *)(v31 + 32);
      if ( (unsigned __int64)(*(_QWORD *)(v31 + 24) - v36) <= 5 )
      {
        v68 = sub_CB6200(v31, " (BB: ", 6u);
        v37 = *(_WORD **)(v68 + 32);
        v31 = v68;
      }
      else
      {
        *(_DWORD *)v36 = 1111631904;
        *(_WORD *)(v36 + 4) = 8250;
        v37 = (_WORD *)(*(_QWORD *)(v31 + 32) + 6LL);
        *(_QWORD *)(v31 + 32) = v37;
      }
      v38 = *(_QWORD *)(v31 + 24) - (_QWORD)v37;
      if ( n > v38 )
      {
        v67 = sub_CB6200(v31, (unsigned __int8 *)src, n);
        v37 = *(_WORD **)(v67 + 32);
        v31 = v67;
        v38 = *(_QWORD *)(v67 + 24) - (_QWORD)v37;
      }
      else if ( n )
      {
        memcpy(v37, src, n);
        v77 = *(_QWORD *)(v31 + 24);
        v37 = (_WORD *)(n + *(_QWORD *)(v31 + 32));
        *(_QWORD *)(v31 + 32) = v37;
        v38 = v77 - (_QWORD)v37;
      }
      if ( v38 <= 5 )
      {
        v66 = sub_CB6200(v31, ", Fn: ", 6u);
        v39 = *(_QWORD **)(v66 + 32);
        v31 = v66;
      }
      else
      {
        *(_DWORD *)v37 = 1850089516;
        v37[2] = 8250;
        v39 = (_QWORD *)(*(_QWORD *)(v31 + 32) + 6LL);
        *(_QWORD *)(v31 + 32) = v39;
      }
      v40 = *(_QWORD *)(v31 + 24) - (_QWORD)v39;
      if ( v23 > v40 )
      {
        v65 = sub_CB6200(v31, v95, v23);
        v39 = *(_QWORD **)(v65 + 32);
        v31 = v65;
        v40 = *(_QWORD *)(v65 + 24) - (_QWORD)v39;
      }
      else if ( v23 )
      {
        memcpy(v39, v95, v23);
        v79 = *(_QWORD *)(v31 + 24);
        v39 = (_QWORD *)(v23 + *(_QWORD *)(v31 + 32));
        *(_QWORD *)(v31 + 32) = v39;
        v40 = v79 - (_QWORD)v39;
      }
      if ( v40 <= 7 )
      {
        v64 = sub_CB6200(v31, ", File: ", 8u);
        v41 = *(_WORD **)(v64 + 32);
        v31 = v64;
      }
      else
      {
        *v39 = 0x203A656C6946202CLL;
        v41 = (_WORD *)(*(_QWORD *)(v31 + 32) + 8LL);
        *(_QWORD *)(v31 + 32) = v41;
      }
      v42 = *(_QWORD *)(v31 + 24) - (_QWORD)v41;
      if ( v42 < a8 )
      {
        v63 = sub_CB6200(v31, a7, a8);
        v41 = *(_WORD **)(v63 + 32);
        v31 = v63;
        v42 = *(_QWORD *)(v63 + 24) - (_QWORD)v41;
      }
      else if ( a8 )
      {
        memcpy(v41, a7, a8);
        v80 = *(_QWORD *)(v31 + 24);
        v41 = (_WORD *)(a8 + *(_QWORD *)(v31 + 32));
        *(_QWORD *)(v31 + 32) = v41;
        v42 = v80 - (_QWORD)v41;
      }
      if ( v42 <= 1 )
      {
        sub_CB6200(v31, (unsigned __int8 *)")\n", 2u);
      }
      else
      {
        *v41 = 2601;
        *(_QWORD *)(v31 + 32) += 2LL;
      }
      goto LABEL_43;
    }
    sub_124B680(v117, "metadata", 8u);
    v118 = 5;
    v119 = "DILocation";
    v120 = 10;
    if ( !(unsigned __int8)sub_C6A630("DILocation", 10, 0) )
    {
      sub_C6B0E0((__int64 *)&v108, (__int64)"DILocation", 0xAu);
      sub_125BB70((__int64)&v112, (__int64)&v108);
      sub_C6BC50(&v118);
      sub_C6A4F0((__int64)&v118, (unsigned __int16 *)&v112);
      sub_C6BC50((unsigned __int16 *)&v112);
      if ( v108 != &v110 )
        j_j___libc_free_0((unsigned __int64)v108);
    }
    sub_124B680(v121, "fn-name", 7u);
    if ( !v95 )
    {
      LOBYTE(v104[0]) = 0;
      v102 = v104;
      v103 = 0;
LABEL_120:
      sub_125BB70((__int64)v122, (__int64)&v102);
      sub_124B680(v123, "bb-name", 7u);
      if ( !src )
      {
        LOBYTE(v107[0]) = 0;
        v105 = v107;
        v106 = 0;
        goto LABEL_126;
      }
      v105 = v107;
      v112 = n;
      if ( n > 0xF )
      {
        v105 = (_QWORD *)sub_22409D0((__int64)&v105, &v112, 0);
        v86 = v105;
        v107[0] = v112;
      }
      else
      {
        if ( n == 1 )
        {
          LOBYTE(v107[0]) = *src;
          v81 = v107;
LABEL_124:
          v106 = n;
          *((_BYTE *)v81 + n) = 0;
LABEL_126:
          v82 = 0;
          sub_125BB70((__int64)v124, (__int64)&v105);
          sub_124B680(v125, "instr", 5u);
          if ( v24 )
            v82 = strlen(v24);
          v126 = 5;
          v127 = v24;
          v128 = v82;
          if ( !(unsigned __int8)sub_C6A630(v24, v82, 0) )
          {
            sub_C6B0E0((__int64 *)&v108, (__int64)v24, v82);
            sub_125BB70((__int64)&v112, (__int64)&v108);
            sub_C6BC50(&v126);
            sub_C6A4F0((__int64)&v126, (unsigned __int16 *)&v112);
            sub_C6BC50((unsigned __int16 *)&v112);
            if ( v108 != &v110 )
              j_j___libc_free_0((unsigned __int64)v108);
          }
          sub_124B680(v129, "action", 6u);
          v131 = "drop";
          v130 = 5;
          v132 = 4;
          if ( !(unsigned __int8)sub_C6A630("drop", 4, 0) )
          {
            sub_C6B0E0((__int64 *)&v108, (__int64)"drop", 4u);
            sub_125BB70((__int64)&v112, (__int64)&v108);
            sub_C6BC50(&v130);
            sub_C6A4F0((__int64)&v130, (unsigned __int16 *)&v112);
            sub_C6BC50((unsigned __int16 *)&v112);
            if ( v108 != &v110 )
              j_j___libc_free_0((unsigned __int64)v108);
          }
          sub_125BF70((__int64)&v108, (__int64)v117, 5);
          v108 = (__int64 *)((char *)v108 + 1);
          LOWORD(v112) = 7;
          v114 = v109;
          v113 = 1;
          v115 = v110;
          v109 = 0;
          v116 = v111;
          v110 = 0;
          v111 = 0;
          v83 = a9[1];
          if ( v83 == a9[2] )
          {
            sub_29C1F00(a9, (__int16 *)a9[1], (__int64)&v112);
          }
          else
          {
            if ( v83 )
            {
              sub_C6A4F0(v83, (unsigned __int16 *)&v112);
              v83 = a9[1];
            }
            a9[1] = v83 + 40;
          }
          v84 = (unsigned __int16 *)v133;
          sub_C6BC50((unsigned __int16 *)&v112);
          sub_C6B900((__int64)&v108);
          sub_C7D6A0(v109, (unsigned __int64)v111 << 6, 8);
          do
          {
            v84 -= 32;
            sub_C6BC50(v84 + 12);
            v85 = *(__m128i **)v84;
            if ( *(_QWORD *)v84 )
            {
              if ( (__m128i *)v85->m128i_i64[0] != &v85[1] )
                j_j___libc_free_0(v85->m128i_i64[0]);
              j_j___libc_free_0((unsigned __int64)v85);
            }
          }
          while ( v84 != (unsigned __int16 *)v117 );
          goto LABEL_111;
        }
        if ( !n )
        {
          v81 = v107;
          goto LABEL_124;
        }
        v86 = v107;
      }
      memcpy(v86, src, n);
      n = v112;
      v81 = v105;
      goto LABEL_124;
    }
    v112 = v23;
    v102 = v104;
    if ( v23 > 0xF )
    {
      v102 = (_QWORD *)sub_22409D0((__int64)&v102, &v112, 0);
      v87 = v102;
      v104[0] = v112;
    }
    else
    {
      if ( v23 == 1 )
      {
        LOBYTE(v104[0]) = *v95;
        v60 = v104;
LABEL_71:
        v103 = v93;
        *((_BYTE *)v60 + v93) = 0;
        goto LABEL_120;
      }
      if ( !v23 )
      {
        v60 = v104;
        goto LABEL_71;
      }
      v87 = v104;
    }
    memcpy(v87, v95, v23);
    v93 = v112;
    v60 = v102;
    goto LABEL_71;
  }
  return 1;
}
