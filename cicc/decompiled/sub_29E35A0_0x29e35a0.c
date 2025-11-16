// Function: sub_29E35A0
// Address: 0x29e35a0
//
void __fastcall sub_29E35A0(__int64 a1, __int64 a2, __int64 a3)
{
  char v3; // r13
  _QWORD *v4; // r12
  unsigned __int64 v5; // r14
  __int64 v6; // rax
  unsigned __int64 v7; // r14
  __int64 v8; // rax
  __int64 v9; // rax
  __int64 v10; // rax
  __int64 v11; // rax
  __int64 v12; // rax
  __int64 v13; // r13
  __int64 v14; // rax
  __int64 v15; // r13
  __int64 v16; // rbx
  unsigned __int64 v17; // r14
  _QWORD *v18; // r12
  int v19; // eax
  unsigned __int64 v20; // rax
  __int64 v21; // rsi
  __int64 v22; // rax
  __int64 v23; // rcx
  unsigned int v24; // edi
  _QWORD *v25; // rdx
  _QWORD *v26; // rsi
  unsigned __int8 *v27; // r11
  int v28; // eax
  unsigned __int64 v29; // rax
  __int64 v30; // rcx
  unsigned __int64 v31; // rax
  unsigned __int64 v32; // rax
  unsigned __int16 v33; // ax
  char v34; // dl
  char *v35; // rax
  __int16 v36; // ax
  int v37; // edx
  int v38; // r9d
  int v39; // edx
  __int64 v40; // rcx
  int v41; // edx
  unsigned int v42; // r15d
  _QWORD *v43; // rax
  _QWORD *v44; // rsi
  __int64 v45; // rax
  __int64 v46; // rcx
  __int64 v47; // rax
  __int64 v48; // rdx
  unsigned __int64 v49; // rdx
  unsigned __int64 v50; // rax
  unsigned __int8 *v51; // r11
  __int64 v52; // rax
  __int64 v53; // rdx
  unsigned __int64 v54; // rdx
  unsigned __int64 v55; // rax
  unsigned __int8 *v56; // r11
  unsigned __int8 v57; // r14
  unsigned __int16 v58; // ax
  __int64 v59; // rax
  unsigned __int8 *v60; // r11
  __int64 v61; // rdx
  unsigned __int8 v62; // al
  unsigned __int64 v63; // rax
  int v64; // eax
  int v65; // edi
  __int64 v66; // rax
  unsigned __int8 *v67; // r11
  __int64 v68; // rax
  __int64 v69; // r15
  __int64 v70; // rax
  char v71; // al
  __int64 v72; // rax
  __int64 v73; // rax
  char v74; // al
  __int64 v75; // rax
  unsigned __int64 v76; // rdx
  char v77; // al
  char v78; // al
  _QWORD *v80; // [rsp+8h] [rbp-198h]
  unsigned __int64 v81; // [rsp+10h] [rbp-190h]
  unsigned __int64 v82; // [rsp+10h] [rbp-190h]
  unsigned __int8 *v83; // [rsp+10h] [rbp-190h]
  __int64 *v84; // [rsp+18h] [rbp-188h]
  unsigned __int8 *v85; // [rsp+20h] [rbp-180h]
  unsigned __int8 *v86; // [rsp+20h] [rbp-180h]
  unsigned __int8 *v87; // [rsp+20h] [rbp-180h]
  unsigned __int8 *v88; // [rsp+20h] [rbp-180h]
  unsigned __int8 *v89; // [rsp+20h] [rbp-180h]
  unsigned __int8 *v90; // [rsp+20h] [rbp-180h]
  unsigned __int8 *v91; // [rsp+20h] [rbp-180h]
  unsigned __int8 *v92; // [rsp+20h] [rbp-180h]
  __int64 v93; // [rsp+28h] [rbp-178h]
  __int64 v94; // [rsp+30h] [rbp-170h]
  __int64 v97; // [rsp+60h] [rbp-140h] BYREF
  unsigned __int64 v98; // [rsp+68h] [rbp-138h] BYREF
  __int64 v99; // [rsp+70h] [rbp-130h] BYREF
  __int64 v100; // [rsp+78h] [rbp-128h] BYREF
  unsigned __int64 v101; // [rsp+80h] [rbp-120h] BYREF
  __int64 v102; // [rsp+88h] [rbp-118h]
  unsigned __int8 *v103; // [rsp+90h] [rbp-110h]
  unsigned int v104; // [rsp+98h] [rbp-108h]
  char v105; // [rsp+A0h] [rbp-100h]
  __int64 *v106; // [rsp+B0h] [rbp-F0h] BYREF
  _BYTE *v107; // [rsp+B8h] [rbp-E8h]
  __int64 v108; // [rsp+C0h] [rbp-E0h]
  _BYTE v109[72]; // [rsp+C8h] [rbp-D8h] BYREF
  __int64 *v110; // [rsp+110h] [rbp-90h] BYREF
  _BYTE *v111; // [rsp+118h] [rbp-88h]
  __int64 v112; // [rsp+120h] [rbp-80h]
  _BYTE v113[120]; // [rsp+128h] [rbp-78h] BYREF

  v4 = (_QWORD *)(a1 + 72);
  v106 = (__int64 *)sub_BD5C60(a1);
  v107 = v109;
  v108 = 0x800000000LL;
  v5 = sub_A74620((_QWORD *)(a1 + 72));
  v6 = *(_QWORD *)(a1 - 32);
  if ( !v6 || *(_BYTE *)v6 || *(_QWORD *)(v6 + 24) != *(_QWORD *)(a1 + 80) )
  {
    if ( !v5 )
      goto LABEL_5;
LABEL_66:
    sub_A77BF0(&v106, v5);
    goto LABEL_5;
  }
  v110 = *(__int64 **)(v6 + 120);
  v32 = sub_A74620(&v110);
  if ( v5 < v32 )
    v5 = v32;
  if ( v5 )
    goto LABEL_66;
LABEL_5:
  v7 = sub_A74640(v4);
  v8 = *(_QWORD *)(a1 - 32);
  if ( !v8 || *(_BYTE *)v8 || *(_QWORD *)(v8 + 24) != *(_QWORD *)(a1 + 80) )
  {
    if ( !v7 )
      goto LABEL_9;
LABEL_62:
    sub_A77C10(&v106, v7);
    goto LABEL_9;
  }
  v110 = *(__int64 **)(v8 + 120);
  v31 = sub_A74640(&v110);
  if ( v7 < v31 )
    v7 = v31;
  if ( v7 )
    goto LABEL_62;
LABEL_9:
  if ( (unsigned __int8)sub_A74710(v4, 0, 22)
    || (v9 = *(_QWORD *)(a1 - 32)) != 0
    && !*(_BYTE *)v9
    && *(_QWORD *)(v9 + 24) == *(_QWORD *)(a1 + 80)
    && (v110 = *(__int64 **)(v9 + 120), (unsigned __int8)sub_A74710(&v110, 0, 22)) )
  {
    sub_A77B20(&v106, 22);
  }
  if ( (unsigned __int8)sub_A74710(v4, 0, 40)
    || (v10 = *(_QWORD *)(a1 - 32)) != 0
    && !*(_BYTE *)v10
    && *(_QWORD *)(v10 + 24) == *(_QWORD *)(a1 + 80)
    && (v110 = *(__int64 **)(v10 + 120), (unsigned __int8)sub_A74710(&v110, 0, 40)) )
  {
    sub_A77B20(&v106, 40);
  }
  v110 = (__int64 *)sub_BD5C60(a1);
  v111 = v113;
  v112 = 0x800000000LL;
  if ( (unsigned __int8)sub_A74710(v4, 0, 43)
    || (v11 = *(_QWORD *)(a1 - 32)) != 0
    && !*(_BYTE *)v11
    && *(_QWORD *)(v11 + 24) == *(_QWORD *)(a1 + 80)
    && (v101 = *(_QWORD *)(v11 + 120), (unsigned __int8)sub_A74710(&v101, 0, 43)) )
  {
    sub_A77B20(&v110, 43);
  }
  if ( (unsigned __int8)sub_A74710(v4, 0, 86)
    || (v12 = *(_QWORD *)(a1 - 32)) != 0
    && !*(_BYTE *)v12
    && *(_QWORD *)(v12 + 24) == *(_QWORD *)(a1 + 80)
    && (v101 = *(_QWORD *)(v12 + 120), (unsigned __int8)sub_A74710(&v101, 0, 86)) )
  {
    v33 = sub_A74820(v4);
    v34 = HIBYTE(v33);
    if ( !HIBYTE(v33) )
    {
      v35 = *(char **)(a1 - 32);
      if ( v35 )
      {
        v34 = *v35;
        if ( *v35 )
        {
          v34 = 0;
        }
        else if ( *((_QWORD *)v35 + 3) == *(_QWORD *)(a1 + 80) )
        {
          v101 = *((_QWORD *)v35 + 15);
          v36 = sub_A74820(&v101);
          v3 = v36;
          v34 = HIBYTE(v36);
        }
      }
      LOBYTE(v33) = v3;
    }
    HIBYTE(v33) = v34;
    sub_A77B90(&v110, v33);
  }
  sub_B492D0((__int64)&v101, a1);
  if ( v105 )
  {
    sub_A78C10(&v110, (__int64)&v101);
    if ( v105 )
    {
      v105 = 0;
      if ( v104 > 0x40 && v103 )
        j_j___libc_free_0_0((unsigned __int64)v103);
      if ( (unsigned int)v102 > 0x40 && v101 )
        j_j___libc_free_0_0(v101);
    }
  }
  if ( (unsigned int)v108 | (unsigned int)v112 )
  {
    v13 = *(_QWORD *)(a1 - 32);
    if ( !v13 || *(_BYTE *)v13 || *(_QWORD *)(a1 + 80) != *(_QWORD *)(v13 + 24) )
    {
      sub_B2BE50(0);
      BUG();
    }
    v84 = (__int64 *)sub_B2BE50(*(_QWORD *)(a1 - 32));
    v14 = v13 + 72;
    if ( *(_QWORD *)(v13 + 80) != v13 + 72 )
    {
      v80 = (_QWORD *)(a1 + 72);
      v15 = *(_QWORD *)(v13 + 80);
      v16 = v14;
      do
      {
        if ( !v15 )
          BUG();
        v17 = *(_QWORD *)(v15 + 24) & 0xFFFFFFFFFFFFFFF8LL;
        if ( v17 == v15 + 24 )
          goto LABEL_152;
        if ( !v17 )
          BUG();
        if ( (unsigned int)*(unsigned __int8 *)(v17 - 24) - 30 > 0xA )
LABEL_152:
          BUG();
        if ( *(_BYTE *)(v17 - 24) == 30 )
        {
          v18 = *(_QWORD **)(v17 - 32LL * (*(_DWORD *)(v17 - 20) & 0x7FFFFFF) - 24);
          v19 = *(unsigned __int8 *)v18;
          if ( (unsigned __int8)v19 > 0x1Cu )
          {
            v20 = (unsigned int)(v19 - 34);
            if ( (unsigned __int8)v20 <= 0x33u )
            {
              v21 = 0x8000000000041LL;
              if ( _bittest64(&v21, v20) )
              {
                v22 = *(unsigned int *)(a2 + 24);
                if ( (_DWORD)v22 )
                {
                  v23 = *(_QWORD *)(a2 + 8);
                  v24 = (v22 - 1) & (((unsigned int)v18 >> 9) ^ ((unsigned int)v18 >> 4));
                  v25 = (_QWORD *)(v23 + ((unsigned __int64)v24 << 6));
                  v26 = (_QWORD *)v25[3];
                  if ( v18 == v26 )
                  {
LABEL_42:
                    if ( v25 != (_QWORD *)(v23 + (v22 << 6)) )
                    {
                      v101 = 6;
                      v102 = 0;
                      v103 = (unsigned __int8 *)v25[7];
                      v27 = v103;
                      if ( v103 != 0 && v103 + 4096 != 0 && v103 != (unsigned __int8 *)-8192LL )
                      {
                        sub_BD6050(&v101, v25[5] & 0xFFFFFFFFFFFFFFF8LL);
                        v27 = v103;
                      }
                      if ( v27 )
                      {
                        v28 = *v27;
                        if ( (unsigned __int8)v28 > 0x1Cu
                          && (v29 = (unsigned int)(v28 - 34), (unsigned __int8)v29 <= 0x33u)
                          && (v30 = 0x8000000000041LL, _bittest64(&v30, v29)) )
                        {
                          if ( v27 != (unsigned __int8 *)-8192LL && v27 != (unsigned __int8 *)-4096LL )
                          {
                            v85 = v27;
                            sub_BD60C0(&v101);
                            v27 = v85;
                          }
                          v39 = *(_DWORD *)(a3 + 56);
                          v40 = *(_QWORD *)(a3 + 40);
                          if ( v39 )
                          {
                            v41 = v39 - 1;
                            v42 = v41 & (((unsigned int)v18 >> 9) ^ ((unsigned int)v18 >> 4));
                            v43 = (_QWORD *)(v40 + 16LL * v42);
                            v44 = (_QWORD *)*v43;
                            if ( v18 == (_QWORD *)*v43 )
                            {
LABEL_98:
                              if ( v27 == (unsigned __int8 *)v43[1] && v18[5] == *(_QWORD *)(v17 + 16) )
                              {
                                v45 = v93;
                                v46 = v94;
                                v86 = v27;
                                LOWORD(v45) = 0;
                                LOWORD(v46) = 0;
                                if ( (unsigned __int8)sub_98CF00(
                                                        v18[4],
                                                        v45,
                                                        v17,
                                                        v46,
                                                        (unsigned int)(qword_5009268 + 1)) )
                                {
                                  v97 = *((_QWORD *)v86 + 9);
                                  v47 = sub_A74DF0((__int64)&v106, 90);
                                  v102 = v48;
                                  v49 = 0;
                                  v101 = v47;
                                  if ( (_BYTE)v102 )
                                    v49 = v101;
                                  v81 = v49;
                                  v50 = sub_A74620(&v97);
                                  v51 = v86;
                                  if ( v50 > v81 )
                                  {
                                    sub_A77390((__int64)&v106, 90);
                                    v51 = v86;
                                  }
                                  v87 = v51;
                                  v52 = sub_A74DF0((__int64)&v106, 91);
                                  v102 = v53;
                                  v54 = 0;
                                  v101 = v52;
                                  if ( (_BYTE)v102 )
                                    v54 = v101;
                                  v82 = v54;
                                  v55 = sub_A74640(&v97);
                                  v56 = v87;
                                  if ( v55 > v82 )
                                  {
                                    sub_A77390((__int64)&v106, 91);
                                    v56 = v87;
                                  }
                                  v83 = v56;
                                  v57 = 0;
                                  v98 = sub_A7B2C0(&v97, v84, 0, (__int64)&v106);
                                  v58 = sub_A74820(&v97);
                                  if ( HIBYTE(v58) )
                                    v57 = v58;
                                  v59 = sub_A74DF0((__int64)&v110, 86);
                                  v60 = v83;
                                  v102 = v61;
                                  v101 = v59;
                                  if ( (_BYTE)v61 && v101 )
                                  {
                                    _BitScanReverse64(&v76, v101);
                                    v62 = 63 - (v76 ^ 0x3F);
                                  }
                                  else
                                  {
                                    v62 = 0;
                                  }
                                  if ( v62 < v57 )
                                  {
                                    sub_A77390((__int64)&v110, 86);
                                    v60 = v83;
                                  }
                                  if ( !(_DWORD)v112 )
                                    goto LABEL_117;
                                  v88 = v60;
                                  v66 = sub_A74D20((__int64)&v110, 97);
                                  v67 = v88;
                                  v99 = v66;
                                  if ( v66 )
                                  {
                                    v68 = sub_A747F0(&v97, 0, 97);
                                    v67 = v88;
                                    v100 = v68;
                                    if ( v68 )
                                    {
                                      v69 = sub_A72AA0(&v99);
                                      v70 = sub_A72AA0(&v100);
                                      sub_AB2160((__int64)&v101, v69, v70, 0);
                                      sub_A78C10(&v110, (__int64)&v101);
                                      v67 = v88;
                                      if ( v104 > 0x40 && v103 )
                                      {
                                        j_j___libc_free_0_0((unsigned __int64)v103);
                                        v67 = v88;
                                      }
                                      if ( (unsigned int)v102 > 0x40 && v101 )
                                      {
                                        v89 = v67;
                                        j_j___libc_free_0_0(v101);
                                        v67 = v89;
                                      }
                                    }
                                  }
                                  v90 = v67;
                                  v71 = sub_A74710(v80, 0, 40);
                                  v60 = v90;
                                  if ( v71 )
                                    goto LABEL_142;
                                  v72 = *(_QWORD *)(a1 - 32);
                                  if ( v72 )
                                  {
                                    if ( !*(_BYTE *)v72 && *(_QWORD *)(v72 + 24) == *(_QWORD *)(a1 + 80) )
                                    {
                                      v101 = *(_QWORD *)(v72 + 120);
                                      v77 = sub_A74710(&v101, 0, 40);
                                      v60 = v90;
                                      if ( v77 )
                                        goto LABEL_142;
                                    }
                                  }
                                  v73 = v18[2];
                                  if ( !v73 )
                                    goto LABEL_117;
                                  if ( !*(_QWORD *)(v73 + 8)
                                    && (v91 = v60, v74 = sub_A74710(v18 + 9, 0, 40), v60 = v91, !v74)
                                    && ((v75 = *(v18 - 4)) == 0
                                     || *(_BYTE *)v75
                                     || *(_QWORD *)(v75 + 24) != v18[10]
                                     || (v101 = *(_QWORD *)(v75 + 120), v78 = sub_A74710(&v101, 0, 40), v60 = v91, !v78)) )
                                  {
LABEL_142:
                                    v92 = v60;
                                    v63 = sub_A7B2C0((__int64 *)&v98, v84, 0, (__int64)&v110);
                                    v60 = v92;
                                  }
                                  else
                                  {
LABEL_117:
                                    v63 = v98;
                                  }
                                  *((_QWORD *)v60 + 9) = v63;
                                }
                              }
                            }
                            else
                            {
                              v64 = 1;
                              while ( v44 != (_QWORD *)-4096LL )
                              {
                                v65 = v64 + 1;
                                v42 = v41 & (v64 + v42);
                                v43 = (_QWORD *)(v40 + 16LL * v42);
                                v44 = (_QWORD *)*v43;
                                if ( v18 == (_QWORD *)*v43 )
                                  goto LABEL_98;
                                v64 = v65;
                              }
                            }
                          }
                        }
                        else if ( v27 != (unsigned __int8 *)-8192LL && v27 != (unsigned __int8 *)-4096LL )
                        {
                          sub_BD60C0(&v101);
                        }
                      }
                    }
                  }
                  else
                  {
                    v37 = 1;
                    while ( v26 != (_QWORD *)-4096LL )
                    {
                      v38 = v37 + 1;
                      v24 = (v22 - 1) & (v37 + v24);
                      v25 = (_QWORD *)(v23 + ((unsigned __int64)v24 << 6));
                      v26 = (_QWORD *)v25[3];
                      if ( v18 == v26 )
                        goto LABEL_42;
                      v37 = v38;
                    }
                  }
                }
              }
            }
          }
        }
        v15 = *(_QWORD *)(v15 + 8);
      }
      while ( v16 != v15 );
    }
  }
  if ( v111 != v113 )
    _libc_free((unsigned __int64)v111);
  if ( v107 != v109 )
    _libc_free((unsigned __int64)v107);
}
