// Function: sub_10890F0
// Address: 0x10890f0
//
__int64 __fastcall sub_10890F0(
        __int64 a1,
        __int64 *a2,
        __int64 a3,
        __int64 a4,
        _QWORD *a5,
        __int64 a6,
        __int64 a7,
        __int64 a8,
        __int64 a9)
{
  __int64 v11; // r13
  char v12; // al
  const char *v15; // rcx
  __int64 v16; // rsi
  const char *v17; // rcx
  __int64 v18; // rax
  __int32 v19; // eax
  unsigned int v20; // esi
  __int64 v21; // r10
  unsigned int v22; // r9d
  __int64 *v23; // rax
  __int64 v24; // rcx
  unsigned int v25; // r9d
  __int64 *v26; // rax
  __int64 v27; // rdi
  const char *v28; // rax
  __int16 v29; // ax
  __int16 v30; // dx
  unsigned __int16 v31; // ax
  int v32; // edi
  int v33; // edi
  __int64 v34; // r11
  unsigned int v35; // esi
  int v36; // eax
  __int64 *v37; // rcx
  __int64 v38; // r9
  __int64 v39; // rdi
  __int64 (*v40)(); // rax
  __m128i *v41; // rsi
  __int64 result; // rax
  __int64 v43; // rdi
  _QWORD *v44; // rax
  __int64 v45; // rdx
  const char *v46; // rax
  _QWORD *v47; // rsi
  int v48; // r15d
  __int64 *v49; // r8
  int v50; // eax
  int v51; // edi
  const char *v52; // rax
  _QWORD *v53; // rax
  __int64 v54; // r15
  __int64 v55; // rax
  unsigned __int64 v56; // rcx
  unsigned __int64 v57; // rax
  unsigned __int64 v58; // rsi
  __int64 v59; // rdi
  char v60; // di
  __int64 v61; // rax
  __int64 v62; // r8
  _QWORD *v63; // rsi
  void *v64; // rax
  __int64 v65; // rdi
  void *v66; // rax
  __m128i v67; // xmm1
  __m128i *v68; // rsi
  _QWORD *v69; // rcx
  _QWORD *v70; // r14
  int v71; // r8d
  int v72; // eax
  int v73; // esi
  int v74; // esi
  __int64 v75; // r9
  int v76; // r10d
  __int64 *v77; // rdx
  __int64 v78; // r15
  __int64 v79; // rdi
  int v80; // eax
  int v81; // eax
  __int64 v82; // rsi
  __int64 v83; // r11
  __int64 v84; // rcx
  int v85; // r10d
  __int64 *v86; // rdx
  int v87; // r9d
  int v88; // r9d
  __int64 v89; // rcx
  int v90; // r10d
  __int64 v91; // r11
  __int64 v92; // rax
  int v93; // eax
  int v94; // eax
  int v95; // r10d
  const char *v96; // [rsp+0h] [rbp-D0h]
  __int64 v97; // [rsp+8h] [rbp-C8h]
  unsigned int v98; // [rsp+8h] [rbp-C8h]
  __int64 v99; // [rsp+18h] [rbp-B8h]
  __int64 v100; // [rsp+20h] [rbp-B0h]
  const char *v101; // [rsp+20h] [rbp-B0h]
  __int64 v103; // [rsp+38h] [rbp-98h] BYREF
  __m128i v104; // [rsp+40h] [rbp-90h] BYREF
  const char *v105; // [rsp+50h] [rbp-80h]
  __int64 v106; // [rsp+58h] [rbp-78h]
  __int16 v107; // [rsp+60h] [rbp-70h]
  __m128i v108; // [rsp+70h] [rbp-60h] BYREF
  const char *v109; // [rsp+80h] [rbp-50h]
  __int16 v110; // [rsp+90h] [rbp-40h]

  v11 = *(_QWORD *)(a7 + 16);
  v12 = *(_BYTE *)(v11 + 8);
  if ( (v12 & 0x10) == 0 )
  {
    v43 = *a2;
    if ( (v12 & 1) != 0 )
    {
      v44 = *(_QWORD **)(v11 - 8);
      v45 = *v44;
      v46 = (const char *)(v44 + 3);
    }
    else
    {
      v45 = 0;
      v46 = 0;
    }
    v105 = v46;
    v104.m128i_i64[0] = (__int64)"symbol '";
    goto LABEL_47;
  }
  if ( (v12 & 2) != 0 )
  {
    v15 = *(const char **)v11;
    if ( !*(_QWORD *)v11 )
    {
      if ( (*(_BYTE *)(v11 + 9) & 0x70) == 0x20 && v12 >= 0 )
      {
        v65 = *(_QWORD *)(v11 + 24);
        v101 = *(const char **)v11;
        *(_BYTE *)(v11 + 8) = v12 | 8;
        v66 = sub_E807D0(v65);
        v15 = v101;
        *(_QWORD *)v11 = v66;
        if ( v66 )
          goto LABEL_4;
        v12 = *(_BYTE *)(v11 + 8);
      }
      v43 = *a2;
      v45 = 0;
      if ( (v12 & 1) != 0 )
      {
        v70 = *(_QWORD **)(v11 - 8);
        v45 = *v70;
        v15 = (const char *)(v70 + 3);
      }
      v105 = v15;
      v104.m128i_i64[0] = (__int64)"assembler label '";
LABEL_47:
      v106 = v45;
      v107 = 1283;
      v47 = *(_QWORD **)(a4 + 16);
      v108.m128i_i64[0] = (__int64)&v104;
      v109 = "' can not be undefined";
      v110 = 770;
      return sub_E66880(v43, v47, (__int64)&v108);
    }
  }
LABEL_4:
  v103 = *(_QWORD *)(a3 + 8);
  v100 = *sub_1085900(a1 + 144, &v103);
  v99 = a8;
  if ( !a8 )
  {
    *a5 = a9;
LABEL_7:
    v105 = 0;
    v104.m128i_i32[1] = 0;
    v19 = sub_E5C2C0((__int64)a2, a3);
    v20 = *(_DWORD *)(a1 + 200);
    v104.m128i_i32[0] = v19;
    if ( (*(_BYTE *)(v11 + 8) & 2) == 0 )
    {
      if ( !v20 )
      {
        ++*(_QWORD *)(a1 + 176);
        goto LABEL_23;
      }
      v21 = *(_QWORD *)(a1 + 184);
LABEL_11:
      v25 = (v20 - 1) & (((unsigned int)v11 >> 9) ^ ((unsigned int)v11 >> 4));
      v26 = (__int64 *)(v21 + 16LL * v25);
      v27 = *v26;
      if ( v11 == *v26 )
      {
LABEL_12:
        v28 = (const char *)v26[1];
LABEL_13:
        v105 = v28;
        goto LABEL_14;
      }
      v71 = 1;
      v37 = 0;
      while ( v27 != -4096 )
      {
        if ( v27 == -8192 && !v37 )
          v37 = v26;
        v93 = v71++;
        v25 = (v20 - 1) & (v93 + v25);
        v26 = (__int64 *)(v21 + 16LL * v25);
        v27 = *v26;
        if ( v11 == *v26 )
          goto LABEL_12;
      }
      if ( !v37 )
        v37 = v26;
      v72 = *(_DWORD *)(a1 + 192);
      ++*(_QWORD *)(a1 + 176);
      v36 = v72 + 1;
      if ( 4 * v36 < 3 * v20 )
      {
        if ( v20 - *(_DWORD *)(a1 + 196) - v36 > v20 >> 3 )
          goto LABEL_25;
        sub_1085690(a1 + 176, v20);
        v73 = *(_DWORD *)(a1 + 200);
        if ( v73 )
        {
          v74 = v73 - 1;
          v75 = *(_QWORD *)(a1 + 184);
          v76 = 1;
          v77 = 0;
          LODWORD(v78) = v74 & (((unsigned int)v11 >> 9) ^ ((unsigned int)v11 >> 4));
          v36 = *(_DWORD *)(a1 + 192) + 1;
          v37 = (__int64 *)(v75 + 16LL * (unsigned int)v78);
          v79 = *v37;
          if ( v11 != *v37 )
          {
            while ( v79 != -4096 )
            {
              if ( v79 == -8192 && !v77 )
                v77 = v37;
              v78 = v74 & (unsigned int)(v78 + v76);
              v37 = (__int64 *)(v75 + 16 * v78);
              v79 = *v37;
              if ( v11 == *v37 )
                goto LABEL_25;
              ++v76;
            }
LABEL_104:
            if ( v77 )
              v37 = v77;
            goto LABEL_25;
          }
          goto LABEL_25;
        }
        goto LABEL_159;
      }
LABEL_23:
      sub_1085690(a1 + 176, 2 * v20);
      v32 = *(_DWORD *)(a1 + 200);
      if ( v32 )
      {
        v33 = v32 - 1;
        v34 = *(_QWORD *)(a1 + 184);
        v35 = v33 & (((unsigned int)v11 >> 9) ^ ((unsigned int)v11 >> 4));
        v36 = *(_DWORD *)(a1 + 192) + 1;
        v37 = (__int64 *)(v34 + 16LL * v35);
        v38 = *v37;
        if ( *v37 != v11 )
        {
          v95 = 1;
          v77 = 0;
          while ( v38 != -4096 )
          {
            if ( !v77 && v38 == -8192 )
              v77 = v37;
            v35 = v33 & (v95 + v35);
            v37 = (__int64 *)(v34 + 16LL * v35);
            v38 = *v37;
            if ( v11 == *v37 )
              goto LABEL_25;
            ++v95;
          }
          goto LABEL_104;
        }
LABEL_25:
        *(_DWORD *)(a1 + 192) = v36;
        if ( *v37 != -4096 )
          --*(_DWORD *)(a1 + 196);
        *v37 = v11;
        v28 = 0;
        v37[1] = 0;
        goto LABEL_13;
      }
LABEL_159:
      ++*(_DWORD *)(a1 + 192);
      BUG();
    }
    if ( v20 )
    {
      v21 = *(_QWORD *)(a1 + 184);
      v22 = (v20 - 1) & (((unsigned int)v11 >> 4) ^ ((unsigned int)v11 >> 9));
      v23 = (__int64 *)(v21 + 16LL * v22);
      v24 = *v23;
      if ( v11 == *v23 )
      {
LABEL_10:
        if ( v23[1] )
          goto LABEL_11;
LABEL_66:
        v52 = *(const char **)v11;
        if ( !*(_QWORD *)v11 )
        {
          if ( (*(_BYTE *)(v11 + 9) & 0x70) != 0x20 || *(char *)(v11 + 8) < 0 )
            BUG();
          *(_BYTE *)(v11 + 8) |= 8u;
          v52 = (const char *)sub_E807D0(*(_QWORD *)(v11 + 24));
          *(_QWORD *)v11 = v52;
        }
        v108.m128i_i64[0] = *((_QWORD *)v52 + 1);
        v53 = sub_1085900(a1 + 144, v108.m128i_i64);
        v54 = *v53;
        v105 = *(const char **)(*v53 + 88LL);
        v55 = *a5 + sub_E5C4C0((__int64)a2, v11);
        *a5 = v55;
        v56 = v55;
        if ( *(_BYTE *)(a1 + 241) && (v57 = *(unsigned int *)(v54 + 128), (_DWORD)v57) && (v58 = v56 >> 20) != 0 )
        {
          v59 = *(_QWORD *)(v54 + 120);
          if ( v58 > v57 )
            v28 = *(const char **)(v59 + 8 * v57 - 8);
          else
            v28 = *(const char **)(v59 + 8 * v58 - 8);
          v105 = v28;
          *a5 = v56 - *((unsigned int *)v28 + 2);
        }
        else
        {
          v28 = v105;
        }
LABEL_14:
        ++*((_DWORD *)v28 + 30);
        v104.m128i_i32[0] += *(_DWORD *)(a4 + 8);
        v29 = (*(__int64 (__fastcall **)(_QWORD, __int64, __int64 *, __int64, bool, __int64))(**(_QWORD **)(*(_QWORD *)a1 + 104LL)
                                                                                            + 32LL))(
                *(_QWORD *)(*(_QWORD *)a1 + 104LL),
                *a2,
                &a7,
                a4,
                v99 != 0,
                a2[1]);
        v30 = *(_WORD *)(a1 + 24);
        v104.m128i_i16[4] = v29;
        switch ( v30 )
        {
          case -31132:
            if ( v29 != 4 )
              goto LABEL_29;
            break;
          case 332:
            if ( v29 != 20 )
              goto LABEL_29;
            break;
          case 452:
            if ( v29 != 10 )
            {
              v31 = v29 - 3;
              goto LABEL_19;
            }
            break;
          case -21916:
            if ( v29 != 17 )
              goto LABEL_29;
            break;
          case -22975:
          case -22962:
            if ( v29 != 17 )
              goto LABEL_40;
            break;
          default:
            goto LABEL_29;
        }
        *a5 += 4LL;
        v30 = *(_WORD *)(a1 + 24);
LABEL_40:
        if ( v30 == 452 )
        {
          v31 = v29 - 3;
          if ( v31 <= 0x12u )
          {
LABEL_19:
            switch ( v31 )
            {
              case 0u:
              case 1u:
              case 5u:
              case 6u:
              case 0xDu:
                BUG();
              case 0xFu:
              case 0x11u:
              case 0x12u:
                *a5 += 4LL;
                break;
              default:
                break;
            }
          }
        }
LABEL_29:
        if ( *(_DWORD *)(a4 + 12) == 11 )
          *a5 = 0;
        v39 = *(_QWORD *)(*(_QWORD *)a1 + 104LL);
        v40 = *(__int64 (**)())(*(_QWORD *)v39 + 40LL);
        if ( v40 == sub_10846C0 || (result = ((__int64 (__fastcall *)(__int64, __int64))v40)(v39, a4), (_BYTE)result) )
        {
          v41 = *(__m128i **)(v100 + 104);
          if ( v41 == *(__m128i **)(v100 + 112) )
          {
            result = sub_1084820((const __m128i **)(v100 + 96), v41, &v104);
          }
          else
          {
            if ( v41 )
            {
              *v41 = _mm_loadu_si128(&v104);
              v41[1].m128i_i64[0] = (__int64)v105;
              v41 = *(__m128i **)(v100 + 104);
            }
            result = v100;
            *(_QWORD *)(v100 + 104) = (char *)v41 + 24;
          }
          if ( *(_WORD *)(a1 + 24) == 358 )
          {
            result = v104.m128i_u16[4];
            if ( v104.m128i_i16[4] == 4 || v104.m128i_i16[4] == 13 )
            {
              v67 = _mm_loadu_si128(&v104);
              v109 = v105;
              v108 = v67;
              v108.m128i_i16[4] = 37;
              v68 = *(__m128i **)(v100 + 104);
              if ( v68 == *(__m128i **)(v100 + 112) )
              {
                return sub_1084820((const __m128i **)(v100 + 96), v68, &v108);
              }
              else
              {
                if ( v68 )
                {
                  *v68 = _mm_loadu_si128(&v108);
                  v68[1].m128i_i64[0] = (__int64)v109;
                  v68 = *(__m128i **)(v100 + 104);
                }
                *(_QWORD *)(v100 + 104) = (char *)v68 + 24;
                return v100;
              }
            }
          }
        }
        return result;
      }
      v48 = 1;
      v49 = 0;
      while ( v24 != -4096 )
      {
        if ( v24 == -8192 && !v49 )
          v49 = v23;
        v94 = v48++;
        v22 = (v20 - 1) & (v94 + v22);
        v23 = (__int64 *)(v21 + 16LL * v22);
        v24 = *v23;
        if ( v11 == *v23 )
          goto LABEL_10;
      }
      if ( !v49 )
        v49 = v23;
      v50 = *(_DWORD *)(a1 + 192);
      ++*(_QWORD *)(a1 + 176);
      v51 = v50 + 1;
      if ( 4 * (v50 + 1) < 3 * v20 )
      {
        if ( v20 - *(_DWORD *)(a1 + 196) - v51 > v20 >> 3 )
        {
LABEL_63:
          *(_DWORD *)(a1 + 192) = v51;
          if ( *v49 != -4096 )
            --*(_DWORD *)(a1 + 196);
          *v49 = v11;
          v49[1] = 0;
          goto LABEL_66;
        }
        v98 = ((unsigned int)v11 >> 4) ^ ((unsigned int)v11 >> 9);
        sub_1085690(a1 + 176, v20);
        v87 = *(_DWORD *)(a1 + 200);
        if ( v87 )
        {
          v88 = v87 - 1;
          v89 = *(_QWORD *)(a1 + 184);
          v90 = 1;
          LODWORD(v91) = v88 & v98;
          v51 = *(_DWORD *)(a1 + 192) + 1;
          v86 = 0;
          v49 = (__int64 *)(v89 + 16LL * (v88 & v98));
          v92 = *v49;
          if ( v11 == *v49 )
            goto LABEL_63;
          while ( v92 != -4096 )
          {
            if ( v92 == -8192 && !v86 )
              v86 = v49;
            v91 = v88 & (unsigned int)(v91 + v90);
            v49 = (__int64 *)(v89 + 16 * v91);
            v92 = *v49;
            if ( v11 == *v49 )
              goto LABEL_63;
            ++v90;
          }
          goto LABEL_113;
        }
        goto LABEL_157;
      }
    }
    else
    {
      ++*(_QWORD *)(a1 + 176);
    }
    sub_1085690(a1 + 176, 2 * v20);
    v80 = *(_DWORD *)(a1 + 200);
    if ( v80 )
    {
      v81 = v80 - 1;
      v82 = *(_QWORD *)(a1 + 184);
      LODWORD(v83) = v81 & (((unsigned int)v11 >> 9) ^ ((unsigned int)v11 >> 4));
      v51 = *(_DWORD *)(a1 + 192) + 1;
      v49 = (__int64 *)(v82 + 16LL * (unsigned int)v83);
      v84 = *v49;
      if ( v11 == *v49 )
        goto LABEL_63;
      v85 = 1;
      v86 = 0;
      while ( v84 != -4096 )
      {
        if ( !v86 && v84 == -8192 )
          v86 = v49;
        v83 = v81 & (unsigned int)(v83 + v85);
        v49 = (__int64 *)(v82 + 16 * v83);
        v84 = *v49;
        if ( v11 == *v49 )
          goto LABEL_63;
        ++v85;
      }
LABEL_113:
      if ( v86 )
        v49 = v86;
      goto LABEL_63;
    }
LABEL_157:
    ++*(_DWORD *)(a1 + 192);
    BUG();
  }
  v16 = *(_QWORD *)(a8 + 16);
  v17 = *(const char **)v16;
  if ( *(_QWORD *)v16 )
  {
LABEL_6:
    v97 = sub_E5C4C0((__int64)a2, v16);
    v18 = sub_E5C2C0((__int64)a2, a3);
    *a5 = a9 + v18 + *(unsigned int *)(a4 + 8) - v97;
    goto LABEL_7;
  }
  v60 = *(_BYTE *)(v16 + 8);
  if ( (*(_BYTE *)(v16 + 9) & 0x70) == 0x20 && v60 >= 0 )
  {
    v96 = *(const char **)v16;
    *(_BYTE *)(v16 + 8) = v60 | 8;
    v64 = sub_E807D0(*(_QWORD *)(v16 + 24));
    v17 = v96;
    *(_QWORD *)v16 = v64;
    if ( v64 )
      goto LABEL_6;
    v60 = *(_BYTE *)(v16 + 8);
  }
  v61 = 0;
  v62 = *a2;
  if ( (v60 & 1) != 0 )
  {
    v69 = *(_QWORD **)(v16 - 8);
    v61 = *v69;
    v17 = (const char *)(v69 + 3);
  }
  v106 = v61;
  v63 = *(_QWORD **)(a4 + 16);
  v105 = v17;
  v107 = 1283;
  v108.m128i_i64[0] = (__int64)&v104;
  v104.m128i_i64[0] = (__int64)"symbol '";
  v109 = "' can not be undefined in a subtraction expression";
  v110 = 770;
  return sub_E66880(v62, v63, (__int64)&v108);
}
