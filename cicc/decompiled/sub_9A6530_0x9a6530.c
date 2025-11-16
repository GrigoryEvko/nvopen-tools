// Function: sub_9A6530
// Address: 0x9a6530
//
__int64 __fastcall sub_9A6530(__int64 a1, __int64 a2, const __m128i *a3, unsigned int a4)
{
  __int64 v4; // r14
  __int64 v5; // r12
  unsigned __int8 v7; // al
  __int64 v8; // r13
  unsigned __int8 v9; // al
  int v10; // ebx
  unsigned int v11; // r13d
  __int64 v12; // rax
  __int64 v13; // rax
  _BYTE *v14; // r15
  unsigned int v15; // r15d
  unsigned int *v16; // rax
  unsigned int v17; // eax
  unsigned __int8 v18; // al
  __int16 v20; // dx
  __int64 v21; // rcx
  __int64 v22; // rdx
  __int64 v23; // rax
  __int64 v24; // r15
  __int64 v25; // rax
  __int64 v26; // rsi
  unsigned int v27; // ecx
  __int64 v28; // rdx
  __int64 v29; // rdi
  __int64 v30; // r8
  __int64 v31; // r15
  __int64 v32; // r13
  __int64 v33; // rdx
  __int64 v34; // rax
  __int64 v35; // r12
  _BYTE *v36; // rdi
  unsigned __int8 *v37; // rsi
  int v38; // eax
  unsigned __int8 *v39; // rax
  int v40; // edx
  int v41; // edx
  unsigned __int8 *v42; // rax
  __int64 v43; // rax
  __int64 v44; // rdx
  __int64 v45; // rax
  unsigned __int8 *v46; // r12
  int v47; // eax
  unsigned __int8 *v48; // rdx
  __int64 v49; // rbx
  char *v50; // r14
  unsigned int v51; // edi
  unsigned __int8 v52; // bl
  unsigned int v53; // eax
  int v54; // eax
  unsigned __int8 *v55; // rdx
  __int64 v56; // rdx
  __int64 v57; // rax
  unsigned int v58; // r14d
  __int64 v59; // rax
  char v60; // al
  __int64 v61; // r8
  unsigned __int8 *v62; // rax
  unsigned __int8 *v63; // rdx
  __int64 v64; // rax
  __int64 v65; // rcx
  _QWORD *v66; // r14
  __int64 v67; // rax
  _QWORD *v68; // rsi
  __int64 v69; // rdi
  bool v70; // zf
  char v71; // al
  _BYTE *v72; // rdi
  __int64 v73; // r14
  __int64 v74; // r15
  unsigned __int8 *v75; // rax
  unsigned __int8 *v76; // rdx
  char v77; // dl
  __int64 v78; // rax
  unsigned __int64 v79; // rdx
  char v80; // dl
  __int64 v81; // rsi
  __int64 v82; // rbx
  int v83; // edx
  int v84; // r9d
  __int64 v85; // rax
  __int64 v86; // rdx
  __int64 v87; // r14
  int v88; // r14d
  __int64 v89; // rax
  __int64 v90; // rdx
  __int64 v91; // [rsp+10h] [rbp-100h]
  __int64 v92; // [rsp+10h] [rbp-100h]
  unsigned int v93; // [rsp+1Ch] [rbp-F4h]
  char *v94; // [rsp+20h] [rbp-F0h]
  unsigned int v95; // [rsp+20h] [rbp-F0h]
  __int64 v96; // [rsp+20h] [rbp-F0h]
  __int64 v97; // [rsp+28h] [rbp-E8h]
  unsigned __int64 v98; // [rsp+28h] [rbp-E8h]
  __int64 v99; // [rsp+30h] [rbp-E0h]
  __int64 v100; // [rsp+30h] [rbp-E0h]
  unsigned __int64 v101; // [rsp+38h] [rbp-D8h]
  __int64 v102; // [rsp+40h] [rbp-D0h]
  __int64 v104; // [rsp+48h] [rbp-C8h]
  _QWORD v105[2]; // [rsp+60h] [rbp-B0h] BYREF
  _BYTE *v106; // [rsp+70h] [rbp-A0h] BYREF
  __int64 v107; // [rsp+78h] [rbp-98h]
  _BYTE v108[32]; // [rsp+80h] [rbp-90h] BYREF
  __int64 v109; // [rsp+A0h] [rbp-70h] BYREF
  unsigned __int8 *v110; // [rsp+A8h] [rbp-68h]
  __int64 v111; // [rsp+B0h] [rbp-60h]
  unsigned int v112; // [rsp+B8h] [rbp-58h]
  char v113; // [rsp+BCh] [rbp-54h]
  unsigned __int8 v114; // [rsp+C0h] [rbp-50h] BYREF

  v4 = a1;
  v5 = a2;
  v7 = *(_BYTE *)a1;
  v8 = *(_QWORD *)(a1 + 8);
  if ( *(_BYTE *)a1 <= 0x15u )
  {
    while ( 1 )
    {
      if ( (unsigned __int8)sub_AC30F0(v4) )
        return 0;
      v9 = *(_BYTE *)v4;
      if ( *(_BYTE *)v4 == 17 )
        return 1;
      if ( *(_BYTE *)(v8 + 8) == 17 )
      {
        v10 = *(_DWORD *)(v8 + 32);
        if ( !v10 )
          return 1;
        v11 = 0;
        while ( 1 )
        {
          v12 = *(_QWORD *)a2;
          if ( *(_DWORD *)(a2 + 8) > 0x40u )
            v12 = *(_QWORD *)(v12 + 8LL * (v11 >> 6));
          if ( (v12 & (1LL << v11)) != 0 )
          {
            v13 = sub_AD69F0(v4, v11);
            v14 = (_BYTE *)v13;
            if ( !v13 || (unsigned __int8)sub_AC30F0(v13) || ((*v14 - 13) & 0xFB) != 0 )
              break;
          }
          if ( ++v11 == v10 )
            return 1;
        }
        return 0;
      }
      if ( v9 != 8 )
        break;
      v4 = *(_QWORD *)(v4 - 128);
      v7 = *(_BYTE *)v4;
      v8 = *(_QWORD *)(v4 + 8);
      if ( *(_BYTE *)v4 > 0x15u )
        goto LABEL_17;
    }
    if ( v9 <= 3u )
    {
      if ( !(unsigned __int8)sub_B2FDA0(v4)
        && (*(_BYTE *)(v4 + 32) & 0xF) != 9
        && !(*(_DWORD *)(*(_QWORD *)(v4 + 8) + 8LL) >> 8) )
      {
        return 1;
      }
      v9 = *(_BYTE *)v4;
    }
    if ( v9 == 5 )
      goto LABEL_20;
    return 0;
  }
LABEL_17:
  if ( v7 != 22 )
    goto LABEL_265;
  sub_B2D8F0(&v109, v4);
  v15 = v114;
  if ( v114 )
  {
    LODWORD(v107) = (_DWORD)v110;
    if ( (unsigned int)v110 > 0x40 )
      sub_C43690(&v106, 0, 0);
    else
      v106 = 0;
    if ( !(unsigned __int8)sub_AB1B10(&v109, &v106) )
    {
      if ( (unsigned int)v107 > 0x40 && v106 )
        j_j___libc_free_0_0(v106);
      if ( v114 )
      {
        v114 = 0;
        if ( v112 > 0x40 && v111 )
          j_j___libc_free_0_0(v111);
        if ( (unsigned int)v110 > 0x40 && v109 )
          j_j___libc_free_0_0(v109);
      }
      return v15;
    }
    if ( (unsigned int)v107 > 0x40 && v106 )
      j_j___libc_free_0_0(v106);
    if ( v114 )
    {
      v114 = 0;
      if ( v112 > 0x40 && v111 )
        j_j___libc_free_0_0(v111);
      if ( (unsigned int)v110 > 0x40 && v109 )
        j_j___libc_free_0_0(v109);
    }
  }
  if ( *(_BYTE *)v4 > 0x15u )
  {
LABEL_265:
    v24 = a3[2].m128i_i64[0];
    if ( v24 && a3[2].m128i_i64[1] )
    {
      if ( !*(_BYTE *)(v24 + 192) )
        sub_CFDFC0(a3[2].m128i_i64[0]);
      v25 = *(unsigned int *)(v24 + 184);
      if ( (_DWORD)v25 )
      {
        v26 = *(_QWORD *)(v24 + 168);
        v27 = (v25 - 1) & (((unsigned int)v4 >> 9) ^ ((unsigned int)v4 >> 4));
        v28 = v26 + 88LL * v27;
        v29 = *(_QWORD *)(v28 + 24);
        if ( v29 == v4 )
        {
LABEL_51:
          if ( v28 == v26 + 88 * v25 )
            goto LABEL_20;
          v30 = *(_QWORD *)(v28 + 40);
          if ( v30 + 32LL * *(unsigned int *)(v28 + 48) == v30 )
            goto LABEL_20;
          v99 = v8;
          v31 = v30 + 32LL * *(unsigned int *)(v28 + 48);
          v32 = *(_QWORD *)(v28 + 40);
          v97 = v5;
          while ( 1 )
          {
            v35 = *(_QWORD *)(v32 + 16);
            if ( !v35 )
              goto LABEL_59;
            v34 = *(unsigned int *)(v32 + 24);
            if ( (_DWORD)v34 == -1 )
              break;
            if ( *(_BYTE *)(*(_QWORD *)(v4 + 8) + 8LL) == 14 )
            {
              v33 = 0;
              if ( *(char *)(v35 + 7) < 0 )
              {
                v33 = sub_BD2BC0(*(_QWORD *)(v32 + 16));
                v34 = *(unsigned int *)(v32 + 24);
              }
              sub_CF90E0(&v109, v35, 16 * v34 + v33);
              if ( (_DWORD)v109 && v111 == v4 )
              {
                if ( (_DWORD)v109 == 43 )
                  goto LABEL_78;
                if ( (_DWORD)v109 == 90 )
                {
                  v44 = *(_QWORD *)(v4 + 8);
                  if ( (unsigned int)*(unsigned __int8 *)(v44 + 8) - 17 <= 1 )
                    v44 = **(_QWORD **)(v44 + 16);
                  v95 = *(_DWORD *)(v44 + 8) >> 8;
                  v45 = sub_B43CB0(a3[2].m128i_i64[1]);
                  if ( !(unsigned __int8)sub_B2F070(v45, v95) )
                  {
LABEL_78:
                    if ( (unsigned __int8)sub_98CF40(v35, a3[2].m128i_i64[1], a3[1].m128i_i64[1], 0) )
                      return 1;
                  }
                }
              }
            }
LABEL_59:
            v32 += 32;
            if ( v31 == v32 )
            {
              v8 = v99;
              v5 = v97;
              goto LABEL_20;
            }
          }
          v36 = *(_BYTE **)(v35 - 32LL * (*(_DWORD *)(v35 + 4) & 0x7FFFFFF));
          if ( *v36 != 82 )
            goto LABEL_59;
          v37 = (unsigned __int8 *)*((_QWORD *)v36 - 8);
          if ( !v37 )
            BUG();
          if ( v37 != (unsigned __int8 *)v4 )
          {
            v38 = *v37;
            if ( (unsigned __int8)v38 > 0x1Cu )
            {
              v54 = v38 - 29;
            }
            else
            {
              if ( (_BYTE)v38 != 5 )
              {
LABEL_67:
                v39 = (unsigned __int8 *)*((_QWORD *)v36 - 4);
                if ( !v39 )
                  goto LABEL_259;
                if ( v39 != (unsigned __int8 *)v4 )
                {
                  v40 = *v39;
                  if ( (unsigned __int8)v40 > 0x1Cu )
                  {
                    v41 = v40 - 29;
                  }
                  else
                  {
                    if ( (_BYTE)v40 != 5 )
                      goto LABEL_59;
                    v41 = *((unsigned __int16 *)v39 + 1);
                  }
                  if ( v41 != 47 )
                    goto LABEL_59;
                  v42 = (v39[7] & 0x40) != 0
                      ? (unsigned __int8 *)*((_QWORD *)v39 - 1)
                      : &v39[-32 * (*((_DWORD *)v39 + 1) & 0x7FFFFFF)];
                  if ( *(_QWORD *)v42 != v4 )
                    goto LABEL_59;
                }
                v94 = (char *)*((_QWORD *)v36 - 8);
                v43 = sub_B53960(v36);
                goto LABEL_77;
              }
              v54 = *((unsigned __int16 *)v37 + 1);
            }
            if ( v54 != 47 )
              goto LABEL_67;
            v55 = (v37[7] & 0x40) != 0
                ? (unsigned __int8 *)*((_QWORD *)v37 - 1)
                : &v37[-32 * (*((_DWORD *)v37 + 1) & 0x7FFFFFF)];
            if ( *(_QWORD *)v55 != v4 )
              goto LABEL_67;
          }
          v94 = (char *)*((_QWORD *)v36 - 4);
          if ( !v94 )
LABEL_259:
            BUG();
          v43 = sub_B53900(v36);
LABEL_77:
          v101 = v43 & 0xFFFFFFFFFFLL | v101 & 0xFFFFFF0000000000LL;
          if ( (unsigned __int8)sub_9867F0(v101, v94) )
            goto LABEL_78;
          goto LABEL_59;
        }
        v83 = 1;
        while ( v29 != -4096 )
        {
          v84 = v83 + 1;
          v27 = (v25 - 1) & (v83 + v27);
          v28 = v26 + 88LL * v27;
          v29 = *(_QWORD *)(v28 + 24);
          if ( v29 == v4 )
            goto LABEL_51;
          v83 = v84;
        }
      }
    }
  }
LABEL_20:
  v16 = (unsigned int *)sub_C94E20(qword_4F862D0);
  if ( v16 )
    v17 = *v16;
  else
    v17 = qword_4F862D0[2];
  if ( a4 >= v17 )
    return 0;
  v18 = *(_BYTE *)v4;
  if ( *(_BYTE *)(v8 + 8) == 14 )
  {
    if ( v18 == 22 )
    {
      if ( (unsigned __int8)sub_B2BAE0(v4)
        && !(unsigned __int8)sub_B2F070(*(_QWORD *)(v4 + 24), *(_DWORD *)(v8 + 8) >> 8)
        || (unsigned __int8)sub_B2F0A0(v4, 1) )
      {
        return 1;
      }
      v18 = *(_BYTE *)v4;
    }
    if ( v18 > 0x1Cu )
    {
      if ( v18 != 63 || (*(_BYTE *)(v4 + 1) & 2) == 0 )
        goto LABEL_26;
    }
    else
    {
      if ( v18 != 5 )
        goto LABEL_28;
      v20 = *(_WORD *)(v4 + 2);
      if ( v20 != 34 )
      {
        if ( v20 == 50 )
        {
          v21 = v4;
LABEL_41:
          v22 = *(_QWORD *)(v21 - 32LL * (*(_DWORD *)(v21 + 4) & 0x7FFFFFF));
          if ( *(_BYTE *)v22 == 3 )
          {
            v23 = *(_QWORD *)(v22 + 24);
            return *(_BYTE *)(v23 + 8) != 16 || *(_QWORD *)(v23 + 32);
          }
          goto LABEL_24;
        }
        goto LABEL_26;
      }
      if ( (*(_BYTE *)(v4 + 1) & 2) == 0 )
        goto LABEL_26;
    }
    v21 = *(_QWORD *)(v4 - 32LL * (*(_DWORD *)(v4 + 4) & 0x7FFFFFF));
    if ( *(_BYTE *)v21 == 5 )
    {
      if ( *(_WORD *)(v21 + 2) != 50 )
        goto LABEL_24;
      goto LABEL_41;
    }
LABEL_26:
    if ( (unsigned __int8)sub_9A4320((unsigned __int8 *)v4, v5, a4 + 1, a3) )
      return 1;
    v18 = *(_BYTE *)v4;
    goto LABEL_28;
  }
LABEL_24:
  if ( v18 == 5 || v18 > 0x1Cu )
    goto LABEL_26;
LABEL_28:
  if ( v18 <= 0x15u )
    return 0;
  v104 = a3[1].m128i_i64[1];
  v102 = a3[2].m128i_i64[1];
  if ( v104 == 0 || v102 == 0 )
    return 0;
  v98 = *(_QWORD *)(v4 + 16);
  if ( !v98 )
    return 0;
  v93 = 0;
  v96 = v4;
  while ( 1 )
  {
    if ( (unsigned int)qword_4F7FF28 <= v93 )
      return 0;
    ++v93;
    v46 = *(unsigned __int8 **)(v98 + 24);
    v47 = *v46;
    if ( *(_BYTE *)(*(_QWORD *)(v96 + 8) + 8LL) != 14 )
      goto LABEL_118;
    if ( (unsigned __int8)(v47 - 34) > 0x33u )
      goto LABEL_119;
    v81 = 0x8000000000041LL;
    if ( _bittest64(&v81, (unsigned int)(v47 - 34))
      && v98 >= (unsigned __int64)&v46[-32 * (*((_DWORD *)v46 + 1) & 0x7FFFFFF)] )
    {
      if ( v47 == 40 )
      {
        v82 = -32 - 32LL * (unsigned int)sub_B491D0(*(_QWORD *)(v98 + 24));
      }
      else
      {
        v82 = -32;
        if ( v47 != 85 )
        {
          if ( v47 != 34 )
            BUG();
          v82 = -96;
        }
      }
      if ( (v46[7] & 0x80u) != 0 )
      {
        v85 = sub_BD2BC0(v46);
        v87 = v85 + v86;
        if ( (v46[7] & 0x80u) == 0 )
        {
          if ( (unsigned int)(v87 >> 4) )
LABEL_261:
            BUG();
        }
        else if ( (unsigned int)((v87 - sub_BD2BC0(v46)) >> 4) )
        {
          if ( (v46[7] & 0x80u) == 0 )
            goto LABEL_261;
          v88 = *(_DWORD *)(sub_BD2BC0(v46) + 8);
          if ( (v46[7] & 0x80u) == 0 )
            BUG();
          v89 = sub_BD2BC0(v46);
          v82 -= 32LL * (unsigned int)(*(_DWORD *)(v89 + v90 - 4) - v88);
        }
      }
      if ( v98 < (unsigned __int64)&v46[v82]
        && (unsigned __int8)sub_B49C40(
                              v46,
                              (__int64)(v98 - (_QWORD)&v46[-32 * (*((_DWORD *)v46 + 1) & 0x7FFFFFF)]) >> 5,
                              0)
        && (unsigned __int8)sub_B19DB0(v104, v46, v102) )
      {
        return 1;
      }
      v47 = *v46;
LABEL_118:
      if ( (unsigned __int8)v47 <= 0x1Cu )
        goto LABEL_119;
    }
    if ( (_BYTE)v47 == 61 || (_BYTE)v47 == 62 )
    {
      v56 = *((_QWORD *)v46 - 4);
      if ( !v56 || v56 != v96 )
      {
LABEL_120:
        if ( (unsigned int)(v47 - 51) > 1 )
          goto LABEL_124;
        goto LABEL_121;
      }
      v57 = *(_QWORD *)(v96 + 8);
      if ( (unsigned int)*(unsigned __int8 *)(v57 + 8) - 17 <= 1 )
        v57 = **(_QWORD **)(v57 + 16);
      v58 = *(_DWORD *)(v57 + 8) >> 8;
      v59 = sub_B43CB0(v46);
      if ( !(unsigned __int8)sub_B2F070(v59, v58) && (unsigned __int8)sub_B19DB0(v104, v46, v102) )
        return 1;
      v47 = *v46;
    }
LABEL_119:
    if ( (unsigned int)(v47 - 48) > 1 )
      goto LABEL_120;
LABEL_121:
    if ( (v46[7] & 0x40) != 0 )
      v48 = (unsigned __int8 *)*((_QWORD *)v46 - 1);
    else
      v48 = &v46[-32 * (*((_DWORD *)v46 + 1) & 0x7FFFFFF)];
    if ( *((_QWORD *)v48 + 4) == v96 && (unsigned __int8)sub_98CF40((__int64)v46, v102, v104, 0) )
      return 1;
LABEL_124:
    if ( *v46 != 82 )
      goto LABEL_133;
    v49 = *((_QWORD *)v46 - 8);
    v50 = (char *)*((_QWORD *)v46 - 4);
    if ( v49 )
    {
      if ( v49 == v96 )
        break;
    }
    if ( (char *)v96 == v50 && v50 && v49 )
    {
      v50 = (char *)*((_QWORD *)v46 - 8);
      v51 = sub_B53960(v46);
      goto LABEL_131;
    }
LABEL_133:
    v98 = *(_QWORD *)(v98 + 8);
    if ( !v98 )
      return 0;
  }
  if ( !v50 )
    goto LABEL_133;
  v51 = sub_B53900(v46);
LABEL_131:
  v52 = sub_9867F0(v51, v50);
  if ( !v52 )
  {
    v53 = sub_B52870(v51);
    if ( !(unsigned __int8)sub_9867F0(v53, v50) )
      goto LABEL_133;
  }
  v109 = 0;
  v106 = v108;
  v107 = 0x400000000LL;
  v110 = &v114;
  v111 = 4;
  v112 = 0;
  v113 = 1;
  v100 = *((_QWORD *)v46 + 2);
  if ( !v100 )
    goto LABEL_133;
  v60 = 1;
LABEL_154:
  v61 = *(_QWORD *)(v100 + 24);
  if ( !v60 )
    goto LABEL_204;
  v62 = v110;
  v63 = &v110[8 * HIDWORD(v111)];
  if ( v110 == v63 )
  {
LABEL_208:
    if ( HIDWORD(v111) >= (unsigned int)v111 )
    {
LABEL_204:
      v91 = *(_QWORD *)(v100 + 24);
      sub_C8CC70(&v109, v91);
      v64 = (unsigned int)v107;
      v61 = v91;
      if ( !v80 )
        goto LABEL_160;
    }
    else
    {
      ++HIDWORD(v111);
      *(_QWORD *)v63 = v61;
      v64 = (unsigned int)v107;
      ++v109;
    }
    if ( v64 + 1 > (unsigned __int64)HIDWORD(v107) )
    {
      v92 = v61;
      sub_C8D5F0(&v106, v108, v64 + 1, 8);
      v64 = (unsigned int)v107;
      v61 = v92;
    }
    *(_QWORD *)&v106[8 * v64] = v61;
    LODWORD(v64) = v107 + 1;
    LODWORD(v107) = v107 + 1;
    goto LABEL_160;
  }
  while ( v61 != *(_QWORD *)v62 )
  {
    v62 += 8;
    if ( v63 == v62 )
      goto LABEL_208;
  }
LABEL_159:
  while ( 2 )
  {
    while ( 2 )
    {
      while ( 2 )
      {
        LODWORD(v64) = v107;
        while ( 1 )
        {
LABEL_160:
          if ( !(_DWORD)v64 )
          {
            v100 = *(_QWORD *)(v100 + 8);
            v60 = v113;
            if ( v100 )
              goto LABEL_154;
            if ( !v113 )
              _libc_free(v110, 0);
            if ( v106 != v108 )
              _libc_free(v106, 0);
            goto LABEL_133;
          }
          v65 = (unsigned int)v64;
          LODWORD(v64) = v64 - 1;
          v66 = *(_QWORD **)&v106[8 * v65 - 8];
          LODWORD(v107) = v64;
          if ( v52 )
            break;
          if ( *(_BYTE *)v66 == 31 )
            goto LABEL_163;
        }
        if ( *(_BYTE *)v66 > 0x1Cu )
        {
          v69 = v66[1];
          if ( (unsigned int)*(unsigned __int8 *)(v69 + 8) - 17 <= 1 )
            v69 = **(_QWORD **)(v69 + 16);
          v70 = (unsigned __int8)sub_BCAC40(v69, 1) == 0;
          v71 = *(_BYTE *)v66;
          if ( v70 )
          {
LABEL_182:
            if ( v71 == 31 )
            {
LABEL_163:
              v67 = *(_QWORD *)((char *)v66 - 32 - 32LL * (v52 ^ 1u));
              v105[0] = v66[5];
              v105[1] = v67;
              if ( (unsigned __int8)sub_B190C0(v105) )
              {
                v68 = v105;
                if ( (unsigned __int8)sub_B19C20(v104, v105, *(_QWORD *)(v102 + 40)) )
                  goto LABEL_165;
              }
              continue;
            }
          }
          else
          {
            if ( v71 == 57 )
              goto LABEL_184;
            if ( v71 != 86 )
              goto LABEL_182;
            if ( *(_QWORD *)(*(v66 - 12) + 8LL) == v66[1] )
            {
              v72 = (_BYTE *)*(v66 - 4);
              if ( *v72 <= 0x15u )
              {
                if ( !(unsigned __int8)sub_AC30F0(v72) )
                {
                  v71 = *(_BYTE *)v66;
                  goto LABEL_182;
                }
LABEL_184:
                v73 = v66[2];
                if ( v73 )
                {
                  v74 = *(_QWORD *)(v73 + 24);
                  if ( v113 )
                  {
LABEL_186:
                    v75 = v110;
                    v76 = &v110[8 * HIDWORD(v111)];
                    if ( v110 == v76 )
                      goto LABEL_197;
                    while ( v74 != *(_QWORD *)v75 )
                    {
                      v75 += 8;
                      if ( v76 == v75 )
                      {
LABEL_197:
                        if ( HIDWORD(v111) < (unsigned int)v111 )
                        {
                          ++HIDWORD(v111);
                          *(_QWORD *)v76 = v74;
                          ++v109;
                          goto LABEL_193;
                        }
                        goto LABEL_192;
                      }
                    }
                    goto LABEL_190;
                  }
                  while ( 1 )
                  {
LABEL_192:
                    sub_C8CC70(&v109, v74);
                    if ( v77 )
                    {
LABEL_193:
                      v78 = (unsigned int)v107;
                      v79 = (unsigned int)v107 + 1LL;
                      if ( v79 > HIDWORD(v107) )
                      {
                        sub_C8D5F0(&v106, v108, v79, 8);
                        v78 = (unsigned int)v107;
                      }
                      *(_QWORD *)&v106[8 * v78] = v74;
                      LODWORD(v107) = v107 + 1;
                      v73 = *(_QWORD *)(v73 + 8);
                      if ( !v73 )
                        goto LABEL_159;
                    }
                    else
                    {
LABEL_190:
                      v73 = *(_QWORD *)(v73 + 8);
                      if ( !v73 )
                        goto LABEL_159;
                    }
                    v74 = *(_QWORD *)(v73 + 24);
                    if ( v113 )
                      goto LABEL_186;
                  }
                }
                continue;
              }
            }
          }
        }
        break;
      }
      if ( !(unsigned __int8)sub_D222C0(v66) )
        continue;
      break;
    }
    v68 = v66;
    if ( !(unsigned __int8)sub_B19DB0(v104, v66, v102) )
      continue;
    break;
  }
LABEL_165:
  if ( !v113 )
    _libc_free(v110, v68);
  if ( v106 != v108 )
    _libc_free(v106, v68);
  return 1;
}
