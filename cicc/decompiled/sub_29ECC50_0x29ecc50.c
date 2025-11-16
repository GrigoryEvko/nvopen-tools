// Function: sub_29ECC50
// Address: 0x29ecc50
//
__int64 __fastcall sub_29ECC50(_QWORD *a1, _QWORD *a2, unsigned __int8 *a3)
{
  __int64 v4; // rax
  int v5; // edx
  __int64 v6; // r12
  __int64 v7; // rax
  __int64 v8; // rdx
  __int64 v9; // rbx
  int v10; // ebx
  __int64 v11; // rax
  __int64 v12; // rdx
  unsigned __int8 *v13; // r13
  unsigned __int8 *v14; // r12
  __int64 v15; // rsi
  __int64 v16; // rcx
  unsigned __int8 *v17; // rbx
  __int64 v18; // r8
  __int64 v19; // r9
  unsigned __int8 **v20; // rdx
  unsigned __int8 **v21; // rax
  __int64 v22; // rax
  __int64 v23; // rbx
  __int64 v24; // r12
  unsigned __int64 v25; // rdi
  char v27; // dl
  __int64 v28; // rdi
  __int64 v29; // rax
  __int64 v30; // rdx
  __int64 v31; // r15
  __int64 v32; // r12
  __int64 v33; // r13
  int v34; // r10d
  unsigned int v35; // edi
  unsigned __int8 **v36; // r14
  unsigned __int8 **v37; // rdx
  unsigned __int8 *v38; // rcx
  __int64 v39; // rdx
  __int64 v40; // rcx
  __int64 v41; // r8
  __int64 v42; // r9
  __int64 v43; // rax
  __int64 v44; // rax
  __m128i *v45; // rsi
  __int64 v46; // r12
  char *v47; // rdi
  unsigned int v48; // edx
  unsigned int v49; // eax
  __int64 v50; // rbx
  __int64 v51; // r13
  __int64 v52; // rsi
  __int64 v53; // r15
  int v54; // r11d
  unsigned int v55; // edi
  _QWORD *v56; // rdx
  _QWORD *v57; // rax
  __int64 v58; // rcx
  __int64 v59; // rdx
  __int64 v60; // rcx
  __int64 v61; // r8
  __int64 v62; // r9
  int v63; // ecx
  int v64; // edi
  __int64 v65; // r8
  unsigned __int8 **v66; // rax
  unsigned __int8 *v67; // rsi
  int v68; // r10d
  __int64 v69; // rdi
  unsigned __int8 *v70; // rsi
  int v71; // r9d
  int v72; // r10d
  __int64 v73; // r8
  unsigned __int8 *v74; // rsi
  int v75; // r9d
  __int64 v76; // rdi
  unsigned __int8 *v77; // rsi
  unsigned __int8 **v78; // [rsp-8h] [rbp-158h]
  int v81; // [rsp+24h] [rbp-12Ch]
  unsigned int v82; // [rsp+28h] [rbp-128h]
  unsigned __int8 *v83; // [rsp+28h] [rbp-128h]
  __int64 v84; // [rsp+30h] [rbp-120h]
  __int64 v85; // [rsp+30h] [rbp-120h]
  __int64 v86; // [rsp+38h] [rbp-118h]
  unsigned __int8 *v87; // [rsp+40h] [rbp-110h]
  unsigned __int8 **v88; // [rsp+48h] [rbp-108h]
  __int64 v89; // [rsp+58h] [rbp-F8h] BYREF
  unsigned __int64 v90; // [rsp+60h] [rbp-F0h] BYREF
  unsigned int v91; // [rsp+68h] [rbp-E8h]
  __m128i v92; // [rsp+70h] [rbp-E0h] BYREF
  __int64 v93; // [rsp+80h] [rbp-D0h] BYREF
  __int64 v94; // [rsp+88h] [rbp-C8h]
  __int64 v95; // [rsp+90h] [rbp-C0h]
  unsigned int v96; // [rsp+98h] [rbp-B8h]
  __int64 v97; // [rsp+A0h] [rbp-B0h] BYREF
  unsigned __int8 **v98; // [rsp+A8h] [rbp-A8h]
  __int64 v99; // [rsp+B0h] [rbp-A0h]
  int v100; // [rsp+B8h] [rbp-98h]
  char v101; // [rsp+BCh] [rbp-94h]
  char v102; // [rsp+C0h] [rbp-90h] BYREF
  __m128i v103; // [rsp+E0h] [rbp-70h] BYREF
  char v104; // [rsp+F0h] [rbp-60h] BYREF

  v4 = sub_B43CC0((__int64)a3);
  v5 = *a3;
  v93 = 0;
  v86 = v4;
  v94 = 0;
  v95 = 0;
  v96 = 0;
  v97 = 0;
  v98 = (unsigned __int8 **)&v102;
  v99 = 4;
  v100 = 0;
  v101 = 1;
  if ( v5 == 40 )
  {
    v6 = -32 - 32LL * (unsigned int)sub_B491D0((__int64)a3);
  }
  else
  {
    v6 = -32;
    if ( v5 != 85 )
    {
      v6 = -96;
      if ( v5 != 34 )
        BUG();
    }
  }
  if ( (a3[7] & 0x80u) != 0 )
  {
    v7 = sub_BD2BC0((__int64)a3);
    v9 = v7 + v8;
    if ( (a3[7] & 0x80u) == 0 )
    {
      if ( !(unsigned int)(v9 >> 4) )
        goto LABEL_9;
    }
    else
    {
      if ( !(unsigned int)((v9 - sub_BD2BC0((__int64)a3)) >> 4) )
        goto LABEL_9;
      if ( (a3[7] & 0x80u) != 0 )
      {
        v10 = *(_DWORD *)(sub_BD2BC0((__int64)a3) + 8);
        if ( (a3[7] & 0x80u) == 0 )
          BUG();
        v11 = sub_BD2BC0((__int64)a3);
        v6 -= 32LL * (unsigned int)(*(_DWORD *)(v11 + v12 - 4) - v10);
        goto LABEL_9;
      }
    }
    BUG();
  }
LABEL_9:
  v87 = &a3[v6];
  v13 = &a3[-32 * (*((_DWORD *)a3 + 1) & 0x7FFFFFF)];
  if ( v13 != v87 )
  {
    v88 = (unsigned __int8 **)v13;
    while ( 1 )
    {
      v14 = *v88;
      v15 = *((_QWORD *)*v88 + 1);
      if ( *(_BYTE *)(v15 + 8) == 14 && *v14 > 0x1Cu )
        break;
LABEL_24:
      v88 += 4;
      if ( v87 == (unsigned __int8 *)v88 )
        goto LABEL_25;
    }
    v91 = sub_AE43F0(v86, v15);
    if ( v91 > 0x40 )
      sub_C43690((__int64)&v90, 0, 0);
    else
      v90 = 0;
    v17 = sub_BD45C0(v14, v86, (__int64)&v90, 1, 0, 0, 0, 0);
    v20 = v78;
    if ( *v17 == 60 )
    {
      if ( v101 )
      {
        v21 = v98;
        v16 = HIDWORD(v99);
        v20 = &v98[HIDWORD(v99)];
        if ( v98 != v20 )
        {
          while ( v17 != *v21 )
          {
            if ( v20 == ++v21 )
              goto LABEL_37;
          }
          goto LABEL_21;
        }
LABEL_37:
        if ( HIDWORD(v99) < (unsigned int)v99 )
        {
          ++HIDWORD(v99);
          *v20 = v17;
          ++v97;
          goto LABEL_39;
        }
      }
      sub_C8CC70((__int64)&v97, (__int64)v17, (__int64)v20, v16, v18, v19);
      if ( v27 )
      {
LABEL_39:
        if ( (v17[7] & 0x20) != 0 )
        {
          v28 = sub_B91C10((__int64)v17, 38);
          if ( v28 )
          {
            v29 = sub_AE94B0(v28);
            v31 = v30;
            v32 = v29;
            if ( v30 != v29 )
            {
              v82 = ((unsigned int)v17 >> 9) ^ ((unsigned int)v17 >> 4);
              while ( 1 )
              {
                while ( 1 )
                {
                  v33 = *(_QWORD *)(v32 + 24);
                  if ( !sub_B10D40(v33 + 48) )
                    break;
                  v32 = *(_QWORD *)(v32 + 8);
                  if ( v32 == v31 )
                    goto LABEL_51;
                }
                if ( !v96 )
                  break;
                v34 = 1;
                v35 = (v96 - 1) & v82;
                v36 = (unsigned __int8 **)(v94 + 88LL * v35);
                v37 = 0;
                v38 = *v36;
                if ( v17 != *v36 )
                {
                  while ( v38 != (unsigned __int8 *)-4096LL )
                  {
                    if ( v38 == (unsigned __int8 *)-8192LL && !v37 )
                      v37 = v36;
                    v35 = (v96 - 1) & (v34 + v35);
                    v36 = (unsigned __int8 **)(v94 + 88LL * v35);
                    v38 = *v36;
                    if ( v17 == *v36 )
                      goto LABEL_47;
                    ++v34;
                  }
                  if ( !v37 )
                    v37 = v36;
                  ++v93;
                  v64 = v95 + 1;
                  if ( 4 * ((int)v95 + 1) < 3 * v96 )
                  {
                    if ( v96 - HIDWORD(v95) - v64 <= v96 >> 3 )
                    {
                      sub_AEC410((__int64)&v93, v96);
                      if ( !v96 )
                      {
LABEL_148:
                        LODWORD(v95) = v95 + 1;
                        BUG();
                      }
                      v72 = 1;
                      LODWORD(v73) = (v96 - 1) & v82;
                      v64 = v95 + 1;
                      v66 = 0;
                      v37 = (unsigned __int8 **)(v94 + 88LL * (unsigned int)v73);
                      v74 = *v37;
                      if ( v17 != *v37 )
                      {
                        while ( v74 != (unsigned __int8 *)-4096LL )
                        {
                          if ( !v66 && v74 == (unsigned __int8 *)-8192LL )
                            v66 = v37;
                          v73 = (v96 - 1) & ((_DWORD)v73 + v72);
                          v37 = (unsigned __int8 **)(v94 + 88 * v73);
                          v74 = *v37;
                          if ( v17 == *v37 )
                            goto LABEL_102;
                          ++v72;
                        }
                        goto LABEL_128;
                      }
                    }
                    goto LABEL_102;
                  }
LABEL_106:
                  sub_AEC410((__int64)&v93, 2 * v96);
                  if ( !v96 )
                    goto LABEL_148;
                  LODWORD(v65) = (v96 - 1) & v82;
                  v64 = v95 + 1;
                  v66 = 0;
                  v37 = (unsigned __int8 **)(v94 + 88LL * (unsigned int)v65);
                  v67 = *v37;
                  if ( v17 != *v37 )
                  {
                    v68 = 1;
                    while ( v67 != (unsigned __int8 *)-4096LL )
                    {
                      if ( v67 == (unsigned __int8 *)-8192LL && !v66 )
                        v66 = v37;
                      v65 = (v96 - 1) & ((_DWORD)v65 + v68);
                      v37 = (unsigned __int8 **)(v94 + 88 * v65);
                      v67 = *v37;
                      if ( v17 == *v37 )
                        goto LABEL_102;
                      ++v68;
                    }
LABEL_128:
                    if ( v66 )
                      v37 = v66;
                  }
LABEL_102:
                  LODWORD(v95) = v64;
                  if ( *v37 != (unsigned __int8 *)-4096LL )
                    --HIDWORD(v95);
                  *v37 = v17;
                  v84 = (__int64)(v37 + 1);
                  v37[5] = (unsigned __int8 *)(v37 + 7);
                  v37[6] = (unsigned __int8 *)0x200000000LL;
                  *(_OWORD *)(v37 + 1) = 0;
                  *(_OWORD *)(v37 + 3) = 0;
                  *(_OWORD *)(v37 + 7) = 0;
                  *(_OWORD *)(v37 + 9) = 0;
                  goto LABEL_48;
                }
LABEL_47:
                v84 = (__int64)(v36 + 1);
LABEL_48:
                v103.m128i_i64[0] = *(_QWORD *)(*(_QWORD *)(v33 + 32 * (1LL - (*(_DWORD *)(v33 + 4) & 0x7FFFFFF))) + 24LL);
                sub_AE7A80((__int64)&v92, v33);
                v103.m128i_i64[1] = sub_B10CD0((__int64)&v92);
                if ( v92.m128i_i64[0] )
                  sub_B91220((__int64)&v92, v92.m128i_i64[0]);
                sub_29EC8C0(v84, &v103, v39, v40, v41, v42);
                v32 = *(_QWORD *)(v32 + 8);
                if ( v32 == v31 )
                  goto LABEL_51;
              }
              ++v93;
              goto LABEL_106;
            }
          }
LABEL_51:
          if ( (v17[7] & 0x20) != 0 )
          {
            v43 = sub_B91C10((__int64)v17, 38);
            if ( v43 )
            {
              v44 = *(_QWORD *)(v43 + 8);
              v45 = (__m128i *)(v44 & 0xFFFFFFFFFFFFFFF8LL);
              if ( (v44 & 4) == 0 )
                v45 = 0;
              sub_B967C0(&v103, v45);
              v46 = v103.m128i_i64[0];
              v47 = (char *)(v103.m128i_i64[0] + 8LL * v103.m128i_u32[2]);
              if ( (char *)v103.m128i_i64[0] != v47 )
              {
                v83 = v17;
                v48 = (unsigned int)v17 >> 4;
                v49 = (unsigned int)v17 >> 9;
                v50 = v103.m128i_i64[0] + 8LL * v103.m128i_u32[2];
                v81 = v49 ^ v48;
                while ( 1 )
                {
                  while ( 1 )
                  {
                    v51 = *(_QWORD *)v46;
                    v52 = *(_QWORD *)(*(_QWORD *)v46 + 24LL);
                    v92.m128i_i64[0] = v52;
                    if ( v52 )
                      sub_B96E90((__int64)&v92, v52, 1);
                    v53 = sub_B10D40((__int64)&v92);
                    if ( v92.m128i_i64[0] )
                      sub_B91220((__int64)&v92, v92.m128i_i64[0]);
                    if ( !v53 )
                      break;
                    v46 += 8;
                    if ( v50 == v46 )
                      goto LABEL_69;
                  }
                  if ( !v96 )
                    break;
                  v54 = 1;
                  v55 = (v96 - 1) & v81;
                  v56 = (_QWORD *)(v94 + 88LL * v55);
                  v57 = 0;
                  v58 = *v56;
                  if ( v83 != (unsigned __int8 *)*v56 )
                  {
                    while ( v58 != -4096 )
                    {
                      if ( v58 == -8192 && !v57 )
                        v57 = v56;
                      v55 = (v96 - 1) & (v54 + v55);
                      v56 = (_QWORD *)(v94 + 88LL * v55);
                      v58 = *v56;
                      if ( v83 == (unsigned __int8 *)*v56 )
                        goto LABEL_65;
                      ++v54;
                    }
                    if ( !v57 )
                      v57 = v56;
                    ++v93;
                    v63 = v95 + 1;
                    if ( 4 * ((int)v95 + 1) < 3 * v96 )
                    {
                      if ( v96 - HIDWORD(v95) - v63 <= v96 >> 3 )
                      {
                        sub_AEC410((__int64)&v93, v96);
                        if ( !v96 )
                        {
LABEL_147:
                          LODWORD(v95) = v95 + 1;
                          BUG();
                        }
                        v75 = 1;
                        LODWORD(v76) = (v96 - 1) & v81;
                        v57 = (_QWORD *)(v94 + 88LL * (unsigned int)v76);
                        v77 = (unsigned __int8 *)*v57;
                        v63 = v95 + 1;
                        if ( v83 != (unsigned __int8 *)*v57 )
                        {
                          while ( v77 != (unsigned __int8 *)-4096LL )
                          {
                            if ( v77 == (unsigned __int8 *)-8192LL && !v53 )
                              v53 = (__int64)v57;
                            v76 = (v96 - 1) & ((_DWORD)v76 + v75);
                            v57 = (_QWORD *)(v94 + 88 * v76);
                            v77 = (unsigned __int8 *)*v57;
                            if ( v83 == (unsigned __int8 *)*v57 )
                              goto LABEL_89;
                            ++v75;
                          }
                          goto LABEL_134;
                        }
                      }
                      goto LABEL_89;
                    }
LABEL_116:
                    sub_AEC410((__int64)&v93, 2 * v96);
                    if ( !v96 )
                      goto LABEL_147;
                    LODWORD(v69) = (v96 - 1) & v81;
                    v57 = (_QWORD *)(v94 + 88LL * (unsigned int)v69);
                    v70 = (unsigned __int8 *)*v57;
                    v63 = v95 + 1;
                    if ( v83 != (unsigned __int8 *)*v57 )
                    {
                      v71 = 1;
                      while ( v70 != (unsigned __int8 *)-4096LL )
                      {
                        if ( v70 == (unsigned __int8 *)-8192LL && !v53 )
                          v53 = (__int64)v57;
                        v69 = (v96 - 1) & ((_DWORD)v69 + v71);
                        v57 = (_QWORD *)(v94 + 88 * v69);
                        v70 = (unsigned __int8 *)*v57;
                        if ( v83 == (unsigned __int8 *)*v57 )
                          goto LABEL_89;
                        ++v71;
                      }
LABEL_134:
                      if ( v53 )
                        v57 = (_QWORD *)v53;
                    }
LABEL_89:
                    LODWORD(v95) = v63;
                    if ( *v57 != -4096 )
                      --HIDWORD(v95);
                    v57[5] = v57 + 7;
                    *v57 = v83;
                    v85 = (__int64)(v57 + 1);
                    v57[6] = 0x200000000LL;
                    *(_OWORD *)(v57 + 1) = 0;
                    *(_OWORD *)(v57 + 3) = 0;
                    *(_OWORD *)(v57 + 7) = 0;
                    *(_OWORD *)(v57 + 9) = 0;
                    goto LABEL_66;
                  }
LABEL_65:
                  v85 = (__int64)(v56 + 1);
LABEL_66:
                  v92.m128i_i64[0] = sub_B12000(v51 + 72);
                  sub_AE7AF0((__int64)&v89, v51);
                  v92.m128i_i64[1] = sub_B10CD0((__int64)&v89);
                  if ( v89 )
                    sub_B91220((__int64)&v89, v89);
                  v46 += 8;
                  sub_29EC8C0(v85, &v92, v59, v60, v61, v62);
                  if ( v50 == v46 )
                  {
LABEL_69:
                    v47 = (char *)v103.m128i_i64[0];
                    goto LABEL_70;
                  }
                }
                ++v93;
                goto LABEL_116;
              }
LABEL_70:
              if ( v47 != &v104 )
                _libc_free((unsigned __int64)v47);
            }
          }
        }
      }
    }
LABEL_21:
    if ( v91 > 0x40 && v90 )
      j_j___libc_free_0_0(v90);
    goto LABEL_24;
  }
LABEL_25:
  if ( !v101 )
    _libc_free((unsigned __int64)v98);
  sub_AE9DC0(a1, a2, (__int64)&v93, v86);
  v22 = v96;
  if ( v96 )
  {
    v23 = v94;
    v24 = v94 + 88LL * v96;
    do
    {
      if ( *(_QWORD *)v23 != -4096 && *(_QWORD *)v23 != -8192 )
      {
        v25 = *(_QWORD *)(v23 + 40);
        if ( v25 != v23 + 56 )
          _libc_free(v25);
        sub_C7D6A0(*(_QWORD *)(v23 + 16), 16LL * *(unsigned int *)(v23 + 32), 8);
      }
      v23 += 88;
    }
    while ( v24 != v23 );
    v22 = v96;
  }
  return sub_C7D6A0(v94, 88 * v22, 8);
}
