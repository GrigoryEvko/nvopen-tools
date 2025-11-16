// Function: sub_F2D1B0
// Address: 0xf2d1b0
//
__int64 __fastcall sub_F2D1B0(__int64 a1)
{
  __int64 v2; // r13
  unsigned int v3; // edx
  __int64 v4; // rax
  __int64 v5; // rsi
  __int64 *v6; // r12
  int v7; // eax
  int v8; // edx
  unsigned int v9; // eax
  __int64 **v10; // rcx
  __int64 *v11; // rdi
  __int64 v12; // r12
  int v13; // eax
  __int64 v14; // rcx
  int v15; // esi
  unsigned int v16; // edx
  __int64 *v17; // rax
  __int64 v18; // rdi
  __int64 v19; // r15
  unsigned int v20; // ebx
  __int64 *v21; // r14
  __int64 v22; // r8
  unsigned __int64 v23; // r9
  unsigned __int64 *v24; // rax
  __int64 v25; // rcx
  unsigned __int64 *v26; // rdx
  __int64 v27; // rax
  __int64 v28; // r14
  __int64 v29; // rcx
  __int64 v30; // r13
  unsigned __int64 *v31; // rdx
  __int64 v32; // rax
  char v33; // dh
  char v34; // si
  __int64 v35; // r13
  __int64 v36; // r15
  int v37; // eax
  __int64 v38; // rdx
  _QWORD *v39; // rax
  _QWORD *i; // rdx
  unsigned __int64 v42; // rdi
  int v43; // ecx
  int v44; // r8d
  unsigned int v45; // ecx
  unsigned int v46; // eax
  _QWORD *v47; // rdi
  int v48; // r12d
  _QWORD *v49; // rax
  unsigned int v50; // r14d
  __int64 v51; // r12
  __int64 v52; // r15
  __int64 v53; // rdi
  unsigned __int64 v54; // r15
  __int64 v55; // rbx
  __int64 v56; // r13
  __int64 v57; // rcx
  __int64 v58; // rsi
  __int64 *v59; // r15
  __int64 v60; // rax
  __int16 v61; // dx
  char v62; // di
  char v63; // si
  unsigned __int8 *v64; // rsi
  int v65; // eax
  int v66; // r8d
  __int64 v67; // rsi
  __int64 v68; // r13
  __int64 v69; // r14
  __int64 v70; // rdx
  char *v71; // rax
  char *v72; // r8
  __int64 v73; // rcx
  char *v74; // rsi
  char *v75; // r9
  _BYTE *v76; // rdx
  char *v77; // r10
  char *v78; // rcx
  _BYTE *v79; // rdi
  _BYTE *v80; // rdi
  _BYTE *v81; // rdi
  _BYTE *v82; // rdi
  unsigned __int64 v83; // rdx
  unsigned __int64 v84; // rax
  _QWORD *v85; // rax
  __int64 v86; // rdx
  _QWORD *j; // rdx
  __int64 *v88; // r13
  __int64 *v89; // r14
  signed __int64 v90; // rcx
  __int64 v91; // [rsp+0h] [rbp-60h]
  __int64 v92; // [rsp+0h] [rbp-60h]
  unsigned __int64 v93; // [rsp+8h] [rbp-58h]
  __int64 v94; // [rsp+10h] [rbp-50h]
  const __m128i *v95; // [rsp+18h] [rbp-48h]
  __int64 v96; // [rsp+18h] [rbp-48h]
  __int64 v97; // [rsp+28h] [rbp-38h] BYREF
  _BYTE v98[48]; // [rsp+30h] [rbp-30h] BYREF

LABEL_1:
  v2 = *(_QWORD *)(a1 + 40);
  v3 = *(_DWORD *)(v2 + 8);
LABEL_2:
  v4 = *(unsigned int *)(v2 + 2136);
  if ( v3 )
  {
    if ( !(_DWORD)v4 )
      goto LABEL_12;
    goto LABEL_4;
  }
  while ( (_DWORD)v4 )
  {
    while ( 1 )
    {
LABEL_4:
      v5 = *(_QWORD *)(v2 + 2104);
      v6 = *(__int64 **)(*(_QWORD *)(v2 + 2128) + 8 * v4 - 8);
      v7 = *(_DWORD *)(v2 + 2120);
      if ( v7 )
      {
        v8 = v7 - 1;
        v9 = (v7 - 1) & (((unsigned int)v6 >> 9) ^ ((unsigned int)v6 >> 4));
        v10 = (__int64 **)(v5 + 8LL * v9);
        v11 = *v10;
        if ( v6 == *v10 )
        {
LABEL_6:
          *v10 = (__int64 *)-8192LL;
          --*(_DWORD *)(v2 + 2112);
          ++*(_DWORD *)(v2 + 2116);
        }
        else
        {
          v43 = 1;
          while ( v11 != (__int64 *)-4096LL )
          {
            v44 = v43 + 1;
            v9 = v8 & (v43 + v9);
            v10 = (__int64 **)(v5 + 8LL * v9);
            v11 = *v10;
            if ( v6 == *v10 )
              goto LABEL_6;
            v43 = v44;
          }
        }
      }
      --*(_DWORD *)(v2 + 2136);
      if ( !v6 )
        break;
      if ( (unsigned __int8)sub_F50EE0(v6, *(_QWORD *)(a1 + 72)) )
        sub_F207A0(a1, v6);
      else
        sub_F15FC0(*(_QWORD *)(a1 + 40), (__int64)v6);
      v2 = *(_QWORD *)(a1 + 40);
      v4 = *(unsigned int *)(v2 + 2136);
      if ( !(_DWORD)v4 )
        goto LABEL_11;
    }
    v2 = *(_QWORD *)(a1 + 40);
LABEL_11:
    v3 = *(_DWORD *)(v2 + 8);
    if ( v3 )
    {
LABEL_12:
      v12 = *(_QWORD *)(*(_QWORD *)v2 + 8LL * v3 - 8);
      v13 = *(_DWORD *)(v2 + 2088);
      *(_DWORD *)(v2 + 8) = v3 - 1;
      v14 = *(_QWORD *)(v2 + 2072);
      if ( v13 )
      {
        v15 = v13 - 1;
        v16 = (v13 - 1) & (((unsigned int)v12 >> 9) ^ ((unsigned int)v12 >> 4));
        v17 = (__int64 *)(v14 + 16LL * v16);
        v18 = *v17;
        if ( *v17 == v12 )
        {
LABEL_14:
          *v17 = -8192;
          --*(_DWORD *)(v2 + 2080);
          ++*(_DWORD *)(v2 + 2084);
        }
        else
        {
          v65 = 1;
          while ( v18 != -4096 )
          {
            v66 = v65 + 1;
            v16 = v15 & (v65 + v16);
            v17 = (__int64 *)(v14 + 16LL * v16);
            v18 = *v17;
            if ( v12 == *v17 )
              goto LABEL_14;
            v65 = v66;
          }
        }
      }
      if ( !v12 )
        goto LABEL_1;
      if ( !(unsigned __int8)sub_F50EE0(v12, *(_QWORD *)(a1 + 72)) )
      {
        if ( !(unsigned __int8)sub_DFE4F0(*(_QWORD *)(a1 + 8)) || !(_BYTE)qword_4F8B208 || byte_4F8B708 )
          goto LABEL_19;
        if ( !byte_4F8B628 )
        {
LABEL_80:
          v94 = *(_QWORD *)(v12 + 40);
          if ( *(_QWORD *)(v12 + 16) )
          {
            v91 = v12;
            v50 = 0;
            v51 = *(_QWORD *)(v12 + 16);
            v96 = a1;
            v52 = 0;
            do
            {
              v55 = *(_QWORD *)(v51 + 24);
              if ( !sub_BD2BE0(v55) )
              {
                if ( (unsigned int)qword_4F8B128 < v50 )
                  goto LABEL_101;
                v56 = *(_QWORD *)(v55 + 40);
                if ( *(_BYTE *)v55 == 84 )
                  v56 = *(_QWORD *)(*(_QWORD *)(v55 - 8)
                                  + 32LL * *(unsigned int *)(v55 + 72)
                                  + 8LL * (unsigned int)((v51 - *(_QWORD *)(v55 - 8)) >> 5));
                if ( v52 && v56 != v52 )
                {
LABEL_101:
                  v12 = v91;
                  a1 = v96;
                  goto LABEL_19;
                }
                if ( !v50 )
                {
                  if ( v94 == v56 )
                    goto LABEL_101;
                  v57 = *(_QWORD *)(v96 + 80);
                  if ( v56 )
                  {
                    v53 = (unsigned int)(*(_DWORD *)(v56 + 44) + 1);
                    if ( (unsigned int)(*(_DWORD *)(v56 + 44) + 1) >= *(_DWORD *)(v57 + 32) )
                      goto LABEL_101;
                  }
                  else
                  {
                    v53 = 0;
                    if ( !*(_DWORD *)(v57 + 32) )
                      goto LABEL_101;
                  }
                  if ( !*(_QWORD *)(*(_QWORD *)(v57 + 24) + 8 * v53) )
                    goto LABEL_101;
                  v54 = *(_QWORD *)(v56 + 48) & 0xFFFFFFFFFFFFFFF8LL;
                  if ( v54 == v56 + 48 )
                    goto LABEL_114;
                  if ( !v54 )
                    BUG();
                  if ( (unsigned int)*(unsigned __int8 *)(v54 - 24) - 30 > 0xA )
                  {
LABEL_114:
                    sub_AA5510(v56);
                  }
                  else if ( v94 != sub_AA5510(v56) && (unsigned int)sub_B46E30(v54 - 24) )
                  {
                    goto LABEL_101;
                  }
                }
                ++v50;
                v52 = v56;
              }
              v51 = *(_QWORD *)(v51 + 8);
            }
            while ( v51 );
            v12 = v91;
            a1 = v96;
            if ( v52 && (unsigned __int8)sub_F1E7F0(v96, v91, v52) )
            {
              *(_BYTE *)(v96 + 240) = 1;
              if ( (*(_BYTE *)(v91 + 7) & 0x40) != 0 )
              {
                v88 = *(__int64 **)(v91 - 8);
                v89 = &v88[4 * (*(_DWORD *)(v91 + 4) & 0x7FFFFFF)];
              }
              else
              {
                v89 = (__int64 *)v91;
                v88 = (__int64 *)(v91 - 32LL * (*(_DWORD *)(v91 + 4) & 0x7FFFFFF));
              }
              while ( v89 != v88 )
              {
                if ( *(_BYTE *)*v88 > 0x1Cu )
                  sub_F15FC0(*(_QWORD *)(v96 + 40), *v88);
                v88 += 4;
              }
            }
          }
          goto LABEL_19;
        }
        if ( *(_BYTE *)v12 == 31 || *(_BYTE *)v12 == 84 )
          goto LABEL_19;
        v70 = 32LL * (*(_DWORD *)(v12 + 4) & 0x7FFFFFF);
        if ( (*(_BYTE *)(v12 + 7) & 0x40) != 0 )
        {
          v71 = *(char **)(v12 - 8);
          v72 = &v71[v70];
        }
        else
        {
          v72 = (char *)v12;
          v71 = (char *)(v12 - v70);
        }
        if ( v70 >> 7 )
        {
          v73 = v70 >> 7 << 7;
          v74 = v71 + 96;
          v75 = v71 + 64;
          v76 = 0;
          v77 = v71 + 32;
          v78 = &v71[v73];
          while ( 1 )
          {
            v82 = (_BYTE *)*((_QWORD *)v74 - 12);
            if ( *v82 <= 0x15u )
            {
              v79 = (_BYTE *)*((_QWORD *)v74 - 8);
              if ( *v79 > 0x15u )
              {
                if ( v76 )
                {
LABEL_160:
                  if ( v79 != v76 )
                  {
                    v71 = v77;
                    goto LABEL_140;
                  }
                }
                else
                {
                  v76 = (_BYTE *)*((_QWORD *)v74 - 8);
                }
                v80 = (_BYTE *)*((_QWORD *)v74 - 4);
                if ( *v80 > 0x15u )
                {
LABEL_163:
                  if ( v80 != v76 )
                  {
                    v71 = v75;
                    goto LABEL_140;
                  }
                }
LABEL_167:
                v81 = *(_BYTE **)v74;
                if ( **(_BYTE **)v74 <= 0x15u )
                  goto LABEL_136;
                goto LABEL_168;
              }
              v80 = (_BYTE *)*((_QWORD *)v74 - 4);
              if ( *v80 > 0x15u )
              {
                if ( v76 )
                  goto LABEL_163;
                v76 = (_BYTE *)*((_QWORD *)v74 - 4);
                goto LABEL_167;
              }
            }
            else
            {
              if ( v76 )
              {
                if ( v82 != v76 )
                  goto LABEL_140;
              }
              else
              {
                v76 = (_BYTE *)*((_QWORD *)v74 - 12);
              }
              v79 = (_BYTE *)*((_QWORD *)v74 - 8);
              if ( *v79 > 0x15u )
                goto LABEL_160;
              v80 = (_BYTE *)*((_QWORD *)v74 - 4);
              if ( *v80 > 0x15u )
                goto LABEL_163;
            }
            v81 = *(_BYTE **)v74;
            if ( **(_BYTE **)v74 <= 0x15u )
              goto LABEL_136;
            if ( !v76 )
            {
              v76 = *(_BYTE **)v74;
              goto LABEL_136;
            }
LABEL_168:
            if ( v76 != v81 )
            {
              v71 = v74;
              goto LABEL_140;
            }
LABEL_136:
            v71 += 128;
            v74 += 128;
            v75 += 128;
            v77 += 128;
            if ( v78 == v71 )
              goto LABEL_179;
          }
        }
        v76 = 0;
LABEL_179:
        v90 = v72 - v71;
        if ( v72 - v71 != 64 )
        {
          if ( v90 != 96 )
          {
            if ( v90 != 32 )
              goto LABEL_80;
LABEL_182:
            if ( **(_BYTE **)v71 <= 0x15u || !v76 || v76 == *(_BYTE **)v71 )
              goto LABEL_80;
LABEL_140:
            if ( v72 == v71 )
              goto LABEL_80;
LABEL_19:
            sub_D5F1F0(*(_QWORD *)(a1 + 32), v12);
            v95 = (const __m128i *)a1;
            v19 = *(_QWORD *)(a1 + 32);
            v20 = 0;
            v97 = 0x1E00000000LL;
            v21 = &v97;
            while ( 1 )
            {
              v22 = *(_QWORD *)(v12 + 48);
              if ( v22 )
                break;
              while ( 1 )
              {
                v21 = (__int64 *)((char *)v21 + 4);
                sub_93FB40(v19, v20);
                if ( v21 == (__int64 *)v98 )
                  goto LABEL_31;
LABEL_27:
                v20 = *(_DWORD *)v21;
                if ( !*(_DWORD *)v21 )
                  break;
                if ( (*(_BYTE *)(v12 + 7) & 0x20) != 0 )
                {
                  v22 = sub_B91C10(v12, v20);
                  if ( v22 )
                    goto LABEL_21;
                }
              }
            }
LABEL_21:
            v23 = *(unsigned int *)(v19 + 8);
            v24 = *(unsigned __int64 **)v19;
            v25 = v23;
            v26 = (unsigned __int64 *)(*(_QWORD *)v19 + 16 * v23);
            if ( *(unsigned __int64 **)v19 == v26 )
            {
LABEL_53:
              v42 = *(unsigned int *)(v19 + 12);
              if ( v23 >= v42 )
              {
                ++v23;
                v93 = v20 | v93 & 0xFFFFFFFF00000000LL;
                if ( v42 < v23 )
                {
                  v92 = v22;
                  sub_C8D5F0(v19, (const void *)(v19 + 16), v23, 0x10u, v22, v23);
                  v22 = v92;
                  v26 = (unsigned __int64 *)(*(_QWORD *)v19 + 16LL * *(unsigned int *)(v19 + 8));
                }
                *v26 = v93;
                v26[1] = v22;
                ++*(_DWORD *)(v19 + 8);
              }
              else
              {
                if ( v26 )
                {
                  *(_DWORD *)v26 = v20;
                  v26[1] = v22;
                  LODWORD(v25) = *(_DWORD *)(v19 + 8);
                }
                v25 = (unsigned int)(v25 + 1);
                *(_DWORD *)(v19 + 8) = v25;
              }
            }
            else
            {
              while ( v20 != *(_DWORD *)v24 )
              {
                v24 += 2;
                if ( v26 == v24 )
                  goto LABEL_53;
              }
              v24[1] = v22;
            }
            v21 = (__int64 *)((char *)v21 + 4);
            if ( v21 == (__int64 *)v98 )
            {
LABEL_31:
              a1 = (__int64)v95;
              v27 = sub_F2CFA0(v95, v12, (__int64)v26, v25, v22, v23);
              v28 = v27;
              if ( v27 )
              {
                if ( v27 == v12 )
                {
                  if ( (unsigned __int8)sub_F50EE0(v12, v95[4].m128i_i64[1]) )
                  {
                    sub_F207A0((__int64)v95, (__int64 *)v12);
                  }
                  else
                  {
                    v68 = *(_QWORD *)(v12 + 16);
                    v69 = v95[2].m128i_i64[1];
                    if ( v68 )
                    {
                      do
                      {
                        sub_F15FC0(v69, *(_QWORD *)(v68 + 24));
                        v68 = *(_QWORD *)(v68 + 8);
                      }
                      while ( v68 );
                      v69 = v95[2].m128i_i64[1];
                    }
                    sub_F15FC0(v69, v12);
                  }
                }
                else
                {
                  if ( *(_QWORD *)(v27 + 48) )
                    goto LABEL_34;
                  v58 = *(_QWORD *)(v12 + 48);
                  v59 = (__int64 *)(v27 + 48);
                  v97 = v58;
                  if ( v58 )
                  {
                    sub_B96E90((__int64)&v97, v58, 1);
                    if ( v59 == &v97 )
                    {
                      if ( v97 )
                        sub_B91220((__int64)&v97, v97);
                      goto LABEL_34;
                    }
                    v67 = *(_QWORD *)(v28 + 48);
                    if ( v67 )
                      sub_B91220((__int64)v59, v67);
                  }
                  else if ( v59 == &v97 )
                  {
                    goto LABEL_34;
                  }
                  v64 = (unsigned __int8 *)v97;
                  *(_QWORD *)(v28 + 48) = v97;
                  if ( v64 )
                    sub_B976B0((__int64)&v97, v64, (__int64)v59);
LABEL_34:
                  LODWORD(v97) = 30;
                  sub_B47C00(v28, v12, (int *)&v97, 1);
                  sub_BD84D0(v12, v28);
                  sub_BD6B90((unsigned __int8 *)v28, (unsigned __int8 *)v12);
                  v29 = 0;
                  v30 = *(_QWORD *)(v12 + 40);
                  v31 = (unsigned __int64 *)(v12 + 24);
                  if ( (*(_BYTE *)v12 == 84) != (*(_BYTE *)v28 == 84) )
                  {
                    if ( *(_BYTE *)v12 == 84 )
                    {
                      v60 = sub_AA5190(v30);
                      v29 = 0;
                      if ( v60 )
                      {
                        v62 = v61;
                        v63 = HIBYTE(v61);
                      }
                      else
                      {
                        v63 = 0;
                        v62 = 0;
                      }
                      v31 = (unsigned __int64 *)v60;
                      LOBYTE(v29) = v62;
                      BYTE1(v29) = v63;
                    }
                    else
                    {
                      v32 = sub_AA4FF0(v30);
                      v34 = 0;
                      v29 = 0;
                      if ( v32 )
                        v34 = v33;
                      LOBYTE(v29) = 1;
                      v31 = (unsigned __int64 *)v32;
                      BYTE1(v29) = v34;
                    }
                  }
                  sub_B44240((_QWORD *)v28, v30, v31, v29);
                  v35 = *(_QWORD *)(v28 + 16);
                  v36 = v95[2].m128i_i64[1];
                  if ( v35 )
                  {
                    do
                    {
                      sub_F15FC0(v36, *(_QWORD *)(v35 + 24));
                      v35 = *(_QWORD *)(v35 + 8);
                    }
                    while ( v35 );
                    v36 = v95[2].m128i_i64[1];
                  }
                  sub_F15FC0(v36, v28);
                  sub_F207A0((__int64)v95, (__int64 *)v12);
                }
                v95[15].m128i_i8[0] = 1;
              }
              goto LABEL_1;
            }
            goto LABEL_27;
          }
          if ( **(_BYTE **)v71 > 0x15u )
          {
            if ( v76 )
            {
              if ( *(_BYTE **)v71 != v76 )
                goto LABEL_140;
            }
            else
            {
              v76 = *(_BYTE **)v71;
            }
          }
          v71 += 32;
        }
        if ( **(_BYTE **)v71 > 0x15u )
        {
          if ( v76 )
          {
            if ( *(_BYTE **)v71 != v76 )
              goto LABEL_140;
          }
          else
          {
            v76 = *(_BYTE **)v71;
          }
        }
        v71 += 32;
        goto LABEL_182;
      }
      sub_F207A0(a1, (__int64 *)v12);
      v2 = *(_QWORD *)(a1 + 40);
      v3 = *(_DWORD *)(v2 + 8);
      goto LABEL_2;
    }
    v4 = *(unsigned int *)(v2 + 2136);
  }
  v37 = *(_DWORD *)(v2 + 2080);
  ++*(_QWORD *)(v2 + 2064);
  if ( v37 )
  {
    v45 = 4 * v37;
    v38 = *(unsigned int *)(v2 + 2088);
    if ( (unsigned int)(4 * v37) < 0x40 )
      v45 = 64;
    if ( v45 >= (unsigned int)v38 )
    {
LABEL_49:
      v39 = *(_QWORD **)(v2 + 2072);
      for ( i = &v39[2 * v38]; i != v39; v39 += 2 )
        *v39 = -4096;
      *(_QWORD *)(v2 + 2080) = 0;
      return *(unsigned __int8 *)(a1 + 240);
    }
    v46 = v37 - 1;
    if ( v46 )
    {
      _BitScanReverse(&v46, v46);
      v47 = *(_QWORD **)(v2 + 2072);
      v48 = 1 << (33 - (v46 ^ 0x1F));
      if ( v48 < 64 )
        v48 = 64;
      if ( (_DWORD)v38 == v48 )
      {
        *(_QWORD *)(v2 + 2080) = 0;
        v49 = &v47[2 * (unsigned int)v38];
        do
        {
          if ( v47 )
            *v47 = -4096;
          v47 += 2;
        }
        while ( v49 != v47 );
        return *(unsigned __int8 *)(a1 + 240);
      }
    }
    else
    {
      v47 = *(_QWORD **)(v2 + 2072);
      v48 = 64;
    }
    sub_C7D6A0((__int64)v47, 16LL * *(unsigned int *)(v2 + 2088), 8);
    v83 = ((((((((4 * v48 / 3u + 1) | ((unsigned __int64)(4 * v48 / 3u + 1) >> 1)) >> 2)
             | (4 * v48 / 3u + 1)
             | ((unsigned __int64)(4 * v48 / 3u + 1) >> 1)) >> 4)
           | (((4 * v48 / 3u + 1) | ((unsigned __int64)(4 * v48 / 3u + 1) >> 1)) >> 2)
           | (4 * v48 / 3u + 1)
           | ((unsigned __int64)(4 * v48 / 3u + 1) >> 1)) >> 8)
         | (((((4 * v48 / 3u + 1) | ((unsigned __int64)(4 * v48 / 3u + 1) >> 1)) >> 2)
           | (4 * v48 / 3u + 1)
           | ((unsigned __int64)(4 * v48 / 3u + 1) >> 1)) >> 4)
         | (((4 * v48 / 3u + 1) | ((unsigned __int64)(4 * v48 / 3u + 1) >> 1)) >> 2)
         | (4 * v48 / 3u + 1)
         | ((unsigned __int64)(4 * v48 / 3u + 1) >> 1)) >> 16;
    v84 = (v83
         | (((((((4 * v48 / 3u + 1) | ((unsigned __int64)(4 * v48 / 3u + 1) >> 1)) >> 2)
             | (4 * v48 / 3u + 1)
             | ((unsigned __int64)(4 * v48 / 3u + 1) >> 1)) >> 4)
           | (((4 * v48 / 3u + 1) | ((unsigned __int64)(4 * v48 / 3u + 1) >> 1)) >> 2)
           | (4 * v48 / 3u + 1)
           | ((unsigned __int64)(4 * v48 / 3u + 1) >> 1)) >> 8)
         | (((((4 * v48 / 3u + 1) | ((unsigned __int64)(4 * v48 / 3u + 1) >> 1)) >> 2)
           | (4 * v48 / 3u + 1)
           | ((unsigned __int64)(4 * v48 / 3u + 1) >> 1)) >> 4)
         | (((4 * v48 / 3u + 1) | ((unsigned __int64)(4 * v48 / 3u + 1) >> 1)) >> 2)
         | (4 * v48 / 3u + 1)
         | ((unsigned __int64)(4 * v48 / 3u + 1) >> 1))
        + 1;
    *(_DWORD *)(v2 + 2088) = v84;
    v85 = (_QWORD *)sub_C7D670(16 * v84, 8);
    v86 = *(unsigned int *)(v2 + 2088);
    *(_QWORD *)(v2 + 2080) = 0;
    *(_QWORD *)(v2 + 2072) = v85;
    for ( j = &v85[2 * v86]; j != v85; v85 += 2 )
    {
      if ( v85 )
        *v85 = -4096;
    }
  }
  else if ( *(_DWORD *)(v2 + 2084) )
  {
    v38 = *(unsigned int *)(v2 + 2088);
    if ( (unsigned int)v38 <= 0x40 )
      goto LABEL_49;
    sub_C7D6A0(*(_QWORD *)(v2 + 2072), 16LL * *(unsigned int *)(v2 + 2088), 8);
    *(_QWORD *)(v2 + 2072) = 0;
    *(_QWORD *)(v2 + 2080) = 0;
    *(_DWORD *)(v2 + 2088) = 0;
  }
  return *(unsigned __int8 *)(a1 + 240);
}
