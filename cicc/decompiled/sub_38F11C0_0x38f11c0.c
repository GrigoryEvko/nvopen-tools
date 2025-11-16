// Function: sub_38F11C0
// Address: 0x38f11c0
//
__int64 __fastcall sub_38F11C0(__int64 a1, __int64 *a2, __int64 *a3)
{
  unsigned int v6; // ecx
  __int64 v7; // r9
  __int64 v8; // r15
  unsigned int *v9; // rax
  __int64 v10; // r8
  const char *v11; // rax
  unsigned int v12; // r14d
  char v14; // al
  unsigned __int64 v15; // rcx
  __int64 v16; // r8
  __int64 v17; // rax
  __m128i v18; // rax
  __m128i v19; // xmm2
  __int16 v20; // r14
  __int64 v21; // rdi
  __int64 v22; // r8
  _DWORD *v23; // rdx
  __int64 (*v24)(); // rax
  __int16 v25; // si
  __int64 v26; // rax
  __int64 v27; // rax
  __int64 v28; // r15
  __m128i v29; // rax
  unsigned __int64 v30; // rax
  __int64 v31; // rdx
  __int64 v32; // r8
  __int64 v33; // rcx
  __int64 v34; // rdx
  __int64 v35; // rdi
  __int16 v36; // r14
  __int64 v37; // r8
  __int64 v38; // rax
  bool v39; // zf
  __int64 v40; // r8
  __int64 v41; // rdx
  __m128i v42; // xmm6
  __m128i v43; // xmm7
  unsigned int v44; // eax
  __int64 v45; // rdx
  __int64 v46; // rax
  __int64 v47; // r14
  char *v48; // r15
  void *v49; // rax
  void *v50; // r14
  __int64 v51; // r8
  __int64 v52; // rax
  __int64 v53; // r14
  __int64 v54; // rax
  __int64 v55; // r14
  __int64 v56; // rdx
  unsigned int v57; // ecx
  __m128i v58; // xmm0
  _DWORD *v59; // rax
  __int64 v60; // rsi
  __int64 v61; // r13
  __int64 v62; // rbx
  __int64 v63; // r15
  __int64 v64; // rax
  __int64 v65; // r12
  __int64 v66; // rdx
  __int64 v67; // rcx
  __int64 v68; // rax
  __int64 v69; // rax
  unsigned __int64 v70; // rsi
  unsigned __int64 v71; // rdi
  unsigned __int64 v72; // rsi
  __int64 v73; // rax
  __int64 v74; // rdx
  unsigned int v75; // ecx
  __int64 v76; // rdi
  __int64 (*v77)(); // r9
  __int64 v78; // rax
  __m128i v79; // xmm5
  __int64 v80; // r14
  __int64 v81; // rdi
  char v82; // al
  unsigned __int64 v83; // rax
  unsigned __int64 v84; // r13
  __int64 v85; // rax
  __int64 v86; // r14
  unsigned __int64 v87; // rdi
  __int64 v88; // rdx
  unsigned __int64 v89; // rax
  unsigned __int64 v90; // rcx
  unsigned __int64 v91; // rax
  unsigned int v92; // [rsp+10h] [rbp-E0h]
  __int64 v93; // [rsp+10h] [rbp-E0h]
  __int64 v94; // [rsp+10h] [rbp-E0h]
  unsigned int v95; // [rsp+18h] [rbp-D8h]
  __int64 v96; // [rsp+18h] [rbp-D8h]
  __int64 v97; // [rsp+18h] [rbp-D8h]
  __int64 v98; // [rsp+18h] [rbp-D8h]
  __int64 v99; // [rsp+18h] [rbp-D8h]
  __int64 v100; // [rsp+18h] [rbp-D8h]
  __int64 v101; // [rsp+18h] [rbp-D8h]
  __m128i v102; // [rsp+20h] [rbp-D0h] BYREF
  __m128i v103; // [rsp+30h] [rbp-C0h] BYREF
  __m128i v104; // [rsp+40h] [rbp-B0h] BYREF
  __int16 v105; // [rsp+50h] [rbp-A0h]
  __m128i v106; // [rsp+60h] [rbp-90h] BYREF
  __m128i v107; // [rsp+70h] [rbp-80h] BYREF
  _BYTE v108[40]; // [rsp+80h] [rbp-70h] BYREF
  __int64 v109; // [rsp+A8h] [rbp-48h]
  __int64 v110; // [rsp+B0h] [rbp-40h]

  v8 = sub_3909290(a1 + 144);
  v9 = *(unsigned int **)(a1 + 152);
  v10 = *v9;
  switch ( (int)v10 )
  {
    case 1:
      return 1;
    case 2:
    case 3:
    case 26:
    case 45:
      v92 = *v9;
      v102 = 0u;
      v14 = sub_38F0EE0(a1, v102.m128i_i64, (unsigned int)v10, v6);
      v16 = v92;
      if ( v14 )
      {
        v59 = (_DWORD *)sub_3909460(a1);
        v16 = v92;
        if ( *v59 == 26 )
        {
          if ( *(_BYTE *)(*(_QWORD *)(a1 + 280) + 32LL) )
          {
            sub_38EB180(a1);
            v80 = sub_38BFA60(*(_QWORD *)(a1 + 320), 1);
            (*(void (__fastcall **)(_QWORD, __int64, _QWORD))(**(_QWORD **)(a1 + 328) + 176LL))(
              *(_QWORD *)(a1 + 328),
              v80,
              0);
            v81 = v80;
            v12 = 0;
            *a2 = sub_38CF310(v81, 0, *(_QWORD *)(a1 + 320), 0);
            *a3 = v8;
          }
          else
          {
            *(_QWORD *)v108 = "invalid token in expression";
            *(_WORD *)&v108[16] = 259;
            return (unsigned int)sub_3909790(a1, v8, v108, 0, 0);
          }
          return v12;
        }
      }
      v17 = *(_QWORD *)(a1 + 336);
      v107 = 0u;
      v106 = 0u;
      if ( *(_BYTE *)(v17 + 359) )
      {
        if ( **(_DWORD **)(a1 + 152) != 17 )
        {
LABEL_10:
          v18 = v102;
          goto LABEL_11;
        }
        sub_38EB180(a1);
        v104 = 0u;
        sub_38F0EE0(a1, v104.m128i_i64, v74, v75);
        *(_WORD *)&v108[16] = 259;
        *(_QWORD *)v108 = "unexpected token in variant, expected ')'";
        v12 = sub_3909E20(a1, 18, v108);
        if ( (_BYTE)v12 )
          return v12;
LABEL_66:
        v58 = _mm_loadu_si128(&v104);
        *(__m128i *)v108 = _mm_loadu_si128(&v102);
        *(__m128i *)&v108[16] = v58;
        v106 = *(__m128i *)v108;
        v107 = v58;
        goto LABEL_10;
      }
      if ( (_DWORD)v16 == 3 )
      {
        if ( **(_DWORD **)(a1 + 152) != 45 )
          goto LABEL_10;
        sub_38EB180(a1);
        v54 = sub_3909290(a1 + 144);
        v104 = 0u;
        v55 = v54;
        if ( (unsigned __int8)sub_38F0EE0(a1, v104.m128i_i64, v56, v57) )
        {
          *(_QWORD *)v108 = "expected symbol variant after '@'";
          *(_WORD *)&v108[16] = 259;
          return (unsigned int)sub_3909790(a1, v55, v108, 0, 0);
        }
        goto LABEL_66;
      }
      v108[0] = 64;
      v69 = sub_16D20C0(v102.m128i_i64, v108, 1u, 0);
      v15 = v69;
      if ( v69 == -1 )
      {
        v15 = v102.m128i_u64[1];
        v18 = v102;
        v71 = 0;
        v72 = 0;
      }
      else
      {
        v70 = v69 + 1;
        v18 = v102;
        if ( v70 > v102.m128i_i64[1] )
          v70 = v102.m128i_u64[1];
        v71 = v102.m128i_i64[1] - v70;
        v72 = v102.m128i_i64[0] + v70;
        if ( v15 && v15 > v102.m128i_i64[1] )
          v15 = v102.m128i_u64[1];
      }
      v106.m128i_i64[0] = v18.m128i_i64[0];
      v106.m128i_i64[1] = v15;
      v107.m128i_i64[0] = v72;
      v107.m128i_i64[1] = v71;
LABEL_11:
      v19 = _mm_loadu_si128(&v102);
      v12 = 1;
      *a3 = v18.m128i_i64[1] + v18.m128i_i64[0];
      v103 = v19;
      if ( !v18.m128i_i64[1] )
        return v12;
      if ( !v107.m128i_i64[1] )
        goto LABEL_77;
      v20 = sub_38CBC00(v107.m128i_i64[0], v107.m128i_i64[1], v18.m128i_i64[1], v15, v16);
      if ( v20 != 1 )
      {
        v103 = _mm_loadu_si128(&v106);
        goto LABEL_15;
      }
      v73 = *(_QWORD *)(a1 + 336);
      if ( *(_BYTE *)(v73 + 172) )
      {
        if ( !*(_BYTE *)(v73 + 359) )
        {
LABEL_77:
          v20 = 0;
LABEL_15:
          v21 = *(_QWORD *)(a1 + 320);
          *(_WORD *)&v108[16] = 261;
          *(_QWORD *)v108 = &v103;
          v22 = sub_38BF510(v21, (__int64)v108);
          if ( (*(_BYTE *)(v22 + 9) & 0xC) == 8 )
          {
            v23 = *(_DWORD **)(v22 + 24);
            if ( *v23 == 4 )
            {
              v24 = *(__int64 (**)())(*((_QWORD *)v23 - 1) + 48LL);
              if ( v24 == sub_2162C30 )
                goto LABEL_18;
              v99 = v22;
              v82 = ((__int64 (__fastcall *)(_DWORD *))v24)(v23 - 2);
              v22 = v99;
              if ( !v82 )
                goto LABEL_18;
              goto LABEL_75;
            }
            if ( *v23 == 1 )
            {
LABEL_75:
              if ( v20 )
              {
                v60 = *a3;
                *(_QWORD *)v108 = "unexpected modifier on variable reference";
                *(_WORD *)&v108[16] = 259;
                return (unsigned int)sub_3909790(a1, v60, v108, 0, 0);
              }
              else
              {
                v12 = 0;
                *a2 = *(_QWORD *)(v22 + 24);
              }
              return v12;
            }
          }
LABEL_18:
          v25 = v20;
          v12 = 0;
          *a2 = sub_38CF310(v22, v25, *(_QWORD *)(a1 + 320), v8);
          return v12;
        }
      }
      v104.m128i_i64[0] = (__int64)"invalid variant '";
      v105 = 1283;
      v104.m128i_i64[1] = (__int64)&v107;
      *(_QWORD *)v108 = &v104;
      *(_WORD *)&v108[16] = 770;
      *(_QWORD *)&v108[8] = "'";
      return (unsigned int)sub_3909790(a1, v107.m128i_i64[0], v108, 0, 0);
    case 4:
      v26 = sub_3909460(a1);
      v96 = sub_39092A0(v26);
      v27 = sub_3909460(a1);
      if ( *(_DWORD *)(v27 + 32) <= 0x40u )
        v28 = *(_QWORD *)(v27 + 24);
      else
        v28 = **(_QWORD **)(v27 + 24);
      v12 = 0;
      *a2 = sub_38CB470(v28, *(_QWORD *)(a1 + 320));
      *a3 = sub_39092B0(*(_QWORD *)(a1 + 152));
      sub_38EB180(a1);
      if ( **(_DWORD **)(a1 + 152) != 2 )
        return v12;
      v29 = *(__m128i *)(sub_3909460(a1) + 8);
      v108[0] = 64;
      v103 = v29;
      v30 = sub_16D20C0(v103.m128i_i64, v108, 1u, 0);
      if ( v30 == -1 )
      {
        v79 = _mm_loadu_si128(&v103);
        v107 = 0u;
        v33 = v103.m128i_i64[1];
        v106 = v79;
        v30 = v79.m128i_u64[1];
      }
      else
      {
        v33 = v103.m128i_i64[1];
        v34 = v30 + 1;
        if ( v30 + 1 > v103.m128i_i64[1] )
          v34 = v103.m128i_i64[1];
        v35 = v103.m128i_i64[1] - v34;
        v31 = v103.m128i_i64[0] + v34;
        if ( v30 && v30 > v103.m128i_i64[1] )
          v30 = v103.m128i_u64[1];
        v106.m128i_i64[0] = v103.m128i_i64[0];
        v106.m128i_i64[1] = v30;
        v107.m128i_i64[0] = v31;
        v107.m128i_i64[1] = v35;
      }
      v36 = 0;
      if ( v33 == v30 )
        goto LABEL_35;
      v36 = sub_38CBC00(v107.m128i_i64[0], v107.m128i_i64[1], v31, v33, v32);
      if ( v36 != 1 )
      {
        v103 = _mm_loadu_si128(&v106);
        v30 = v103.m128i_u64[1];
LABEL_35:
        if ( v30 == 1 && (*(_BYTE *)v103.m128i_i64[0] & 0xFB) == 0x62 )
        {
          v93 = sub_38C4F00(*(_QWORD *)(a1 + 320), v28, *(_BYTE *)v103.m128i_i64[0] == 98);
          v38 = sub_38CF310(v93, v36, *(_QWORD *)(a1 + 320), 0);
          v39 = v103.m128i_i64[1] == 1;
          v40 = v93;
          *a2 = v38;
          if ( v39 && *(_BYTE *)v103.m128i_i64[0] == 98 && (*(_QWORD *)v93 & 0xFFFFFFFFFFFFFFF8LL) == 0 )
          {
            if ( (*(_BYTE *)(v93 + 9) & 0xC) != 8
              || (*(_BYTE *)(v93 + 8) |= 4u,
                  v91 = (unsigned __int64)sub_38CE440(*(_QWORD *)(v93 + 24)),
                  v40 = v93,
                  *(_QWORD *)v93 = v91 | *(_QWORD *)v93 & 7LL,
                  !v91) )
            {
              *(_QWORD *)v108 = "directional label undefined";
              *(_WORD *)&v108[16] = 259;
              return (unsigned int)sub_3909790(a1, v96, v108, 0, 0);
            }
          }
          v41 = *(unsigned int *)(a1 + 612);
          v42 = _mm_loadu_si128((const __m128i *)(a1 + 560));
          v43 = _mm_loadu_si128((const __m128i *)(a1 + 576));
          v109 = *(_QWORD *)(a1 + 592);
          *(__m128i *)&v108[8] = v42;
          v110 = v96;
          v44 = *(_DWORD *)(a1 + 608);
          *(__m128i *)&v108[24] = v43;
          if ( v44 >= (unsigned int)v41 )
          {
            v100 = v40;
            v83 = (((((v41 + 2) | ((unsigned __int64)(v41 + 2) >> 1)) >> 2)
                  | (v41 + 2)
                  | ((unsigned __int64)(v41 + 2) >> 1)) >> 4)
                | (((v41 + 2) | ((unsigned __int64)(v41 + 2) >> 1)) >> 2)
                | (v41 + 2)
                | ((unsigned __int64)(v41 + 2) >> 1);
            v84 = ((v83 >> 8) | v83 | (((v83 >> 8) | v83) >> 16) | (((v83 >> 8) | v83) >> 32)) + 1;
            if ( v84 > 0xFFFFFFFF )
              v84 = 0xFFFFFFFFLL;
            v85 = malloc(56 * v84);
            v40 = v100;
            v86 = v85;
            if ( !v85 )
            {
              sub_16BD1C0("Allocation failed", 1u);
              v40 = v100;
            }
            v87 = *(_QWORD *)(a1 + 600);
            v88 = v86;
            v89 = v87;
            v90 = v87 + 56LL * *(unsigned int *)(a1 + 608);
            if ( v87 != v90 )
            {
              do
              {
                if ( v88 )
                {
                  *(_QWORD *)v88 = *(_QWORD *)v89;
                  *(__m128i *)(v88 + 8) = _mm_loadu_si128((const __m128i *)(v89 + 8));
                  *(__m128i *)(v88 + 24) = _mm_loadu_si128((const __m128i *)(v89 + 24));
                  *(_QWORD *)(v88 + 40) = *(_QWORD *)(v89 + 40);
                  *(_QWORD *)(v88 + 48) = *(_QWORD *)(v89 + 48);
                }
                v89 += 56LL;
                v88 += 56;
              }
              while ( v90 != v89 );
            }
            if ( v87 != a1 + 616 )
            {
              v101 = v40;
              _libc_free(v87);
              v40 = v101;
            }
            *(_QWORD *)(a1 + 600) = v86;
            v44 = *(_DWORD *)(a1 + 608);
            *(_DWORD *)(a1 + 612) = v84;
          }
          v45 = *(_QWORD *)(a1 + 600) + 56LL * v44;
          if ( v45 )
          {
            *(_QWORD *)v45 = v40;
            *(__m128i *)(v45 + 8) = _mm_loadu_si128((const __m128i *)&v108[8]);
            *(__m128i *)(v45 + 24) = _mm_loadu_si128((const __m128i *)&v108[24]);
            *(_QWORD *)(v45 + 40) = v109;
            *(_QWORD *)(v45 + 48) = v110;
            v44 = *(_DWORD *)(a1 + 608);
          }
          *(_DWORD *)(a1 + 608) = v44 + 1;
LABEL_60:
          *a3 = sub_39092B0(*(_QWORD *)(a1 + 152));
          sub_38EB180(a1);
        }
        return 0;
      }
      v105 = 1283;
      v104.m128i_i64[0] = (__int64)"invalid variant '";
      v104.m128i_i64[1] = (__int64)&v107;
      *(_QWORD *)v108 = &v104;
      *(_WORD *)&v108[16] = 770;
      *(_QWORD *)&v108[8] = "'";
      return (unsigned int)sub_3909CF0(a1, v108, 0, 0, v37, v108);
    case 5:
      v108[17] = 1;
      v11 = "literal value out of range for directive";
      goto LABEL_5;
    case 6:
      v46 = sub_3909460(a1);
      v47 = *(_QWORD *)(v46 + 16);
      v48 = *(char **)(v46 + 8);
      v49 = sub_1698280();
      sub_169E660((__int64)v108, v49, v48, v47);
      v50 = sub_16982C0();
      if ( *(void **)&v108[8] == v50 )
        sub_169D930((__int64)&v106, (__int64)&v108[8]);
      else
        sub_169D7E0((__int64)&v106, (__int64 *)&v108[8]);
      v51 = v106.m128i_i64[0];
      if ( v106.m128i_i32[2] > 0x40u )
      {
        v97 = *(_QWORD *)v106.m128i_i64[0];
        j_j___libc_free_0_0(v106.m128i_u64[0]);
        v51 = v97;
      }
      *a2 = sub_38CB470(v51, *(_QWORD *)(a1 + 320));
      *a3 = sub_39092B0(*(_QWORD *)(a1 + 152));
      sub_38EB180(a1);
      if ( v50 == *(void **)&v108[8] )
      {
        v61 = *(_QWORD *)&v108[16];
        if ( *(_QWORD *)&v108[16] )
        {
          v62 = *(_QWORD *)&v108[16] + 32LL * *(_QWORD *)(*(_QWORD *)&v108[16] - 8LL);
          if ( *(_QWORD *)&v108[16] != v62 )
          {
            do
            {
              v62 -= 32;
              if ( v50 == *(void **)(v62 + 8) )
              {
                v63 = *(_QWORD *)(v62 + 16);
                if ( v63 )
                {
                  v64 = 32LL * *(_QWORD *)(v63 - 8);
                  v65 = v63 + v64;
                  while ( v63 != v65 )
                  {
                    v65 -= 32;
                    if ( v50 == *(void **)(v65 + 8) )
                    {
                      v66 = *(_QWORD *)(v65 + 16);
                      if ( v66 )
                      {
                        v67 = 32LL * *(_QWORD *)(v66 - 8);
                        v68 = v66 + v67;
                        if ( v66 != v66 + v67 )
                        {
                          do
                          {
                            v94 = v66;
                            v98 = v68 - 32;
                            sub_127D120((_QWORD *)(v68 - 24));
                            v68 = v98;
                            v66 = v94;
                          }
                          while ( v94 != v98 );
                        }
                        j_j_j___libc_free_0_0(v66 - 8);
                      }
                    }
                    else
                    {
                      sub_1698460(v65 + 8);
                    }
                  }
                  j_j_j___libc_free_0_0(v63 - 8);
                }
              }
              else
              {
                sub_1698460(v62 + 8);
              }
            }
            while ( v61 != v62 );
          }
          j_j_j___libc_free_0_0(v61 - 8);
        }
      }
      else
      {
        sub_1698460((__int64)&v108[8]);
      }
      return 0;
    case 12:
      sub_38EB180(a1);
      v12 = sub_38F11C0(a1, a2, a3);
      if ( (_BYTE)v12 )
        return 1;
      *a2 = sub_38CB340(3, *a2, *(_QWORD *)(a1 + 320), v8);
      return v12;
    case 13:
      sub_38EB180(a1);
      v12 = sub_38F11C0(a1, a2, a3);
      if ( (_BYTE)v12 )
        return 1;
      *a2 = sub_38CB340(1, *a2, *(_QWORD *)(a1 + 320), v8);
      return v12;
    case 14:
      sub_38EB180(a1);
      v12 = sub_38F11C0(a1, a2, a3);
      if ( (_BYTE)v12 )
        return 1;
      *a2 = sub_38CB340(2, *a2, *(_QWORD *)(a1 + 320), v8);
      return v12;
    case 17:
      sub_38EB180(a1);
      return sub_38ECD60(a1, a2, a3);
    case 19:
      if ( !*(_BYTE *)(*(_QWORD *)(a1 + 368) + 16LL) )
      {
        v108[17] = 1;
        v11 = "brackets expression not supported on this target";
        goto LABEL_5;
      }
      sub_38EB180(a1);
      *(_QWORD *)v108 = 0;
      if ( sub_38EB6A0(a1, a2, (__int64)v108) )
        return 1;
      v52 = sub_3909460(a1);
      *a3 = sub_39092B0(v52);
      *(_QWORD *)v108 = "expected ']' in brackets expression";
      *(_WORD *)&v108[16] = 259;
      return (unsigned int)sub_3909E20(a1, 20, v108);
    case 24:
      v53 = sub_38BFA60(*(_QWORD *)(a1 + 320), 1);
      (*(void (__fastcall **)(_QWORD, __int64, _QWORD))(**(_QWORD **)(a1 + 328) + 176LL))(*(_QWORD *)(a1 + 328), v53, 0);
      *a2 = sub_38CF310(v53, 0, *(_QWORD *)(a1 + 320), 0);
      goto LABEL_60;
    case 34:
      sub_38EB180(a1);
      v12 = sub_38F11C0(a1, a2, a3);
      if ( (_BYTE)v12 )
        return 1;
      *a2 = sub_38CB340(0, *a2, *(_QWORD *)(a1 + 320), v8);
      return v12;
    case 46:
    case 47:
    case 48:
    case 49:
    case 50:
    case 51:
    case 52:
    case 53:
    case 54:
    case 55:
    case 56:
    case 57:
    case 58:
    case 59:
    case 60:
    case 61:
    case 62:
    case 63:
    case 64:
    case 65:
    case 66:
    case 67:
    case 68:
    case 69:
      v95 = *v9;
      sub_38EB180(a1);
      v10 = v95;
      if ( **(_DWORD **)(a1 + 152) != 17 )
      {
        v108[17] = 1;
        v11 = "expected '(' after operator";
        goto LABEL_5;
      }
      sub_38EB180(a1);
      if ( !sub_38EB6A0(a1, a2, (__int64)a3) )
      {
        v10 = v95;
        if ( **(_DWORD **)(a1 + 152) == 18 )
        {
          v12 = 1;
          sub_38EB180(a1);
          v76 = *(_QWORD *)(a1 + 8);
          v77 = *(__int64 (**)())(*(_QWORD *)v76 + 176LL);
          v78 = 0;
          if ( v77 != sub_38E2A40 )
          {
            v78 = ((__int64 (__fastcall *)(__int64, __int64, _QWORD, _QWORD))v77)(v76, *a2, v95, *(_QWORD *)(a1 + 320));
            LOBYTE(v12) = v78 == 0;
          }
          *a2 = v78;
          return v12;
        }
        v108[17] = 1;
        v11 = "expected ')'";
LABEL_5:
        *(_QWORD *)v108 = v11;
        v108[16] = 3;
        return (unsigned int)sub_3909CF0(a1, v108, 0, 0, v10, v7);
      }
      return 1;
    default:
      v108[17] = 1;
      v11 = "unknown token in expression";
      goto LABEL_5;
  }
}
