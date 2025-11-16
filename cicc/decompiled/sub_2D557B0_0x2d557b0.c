// Function: sub_2D557B0
// Address: 0x2d557b0
//
__int64 __fastcall sub_2D557B0(__int64 *a1, __int64 a2, __int64 a3)
{
  __int64 v4; // rdi
  unsigned int v5; // r13d
  __int64 v6; // r15
  unsigned __int8 *v7; // rax
  size_t v8; // rdx
  __int64 v9; // rcx
  __int64 v10; // r9
  __int64 v11; // r14
  __int64 v12; // rdx
  __int64 v13; // rbx
  __int64 v14; // rax
  __int64 v15; // r8
  __int64 *v16; // rdi
  __int64 v17; // r14
  char *v18; // rax
  __int64 v19; // rax
  __int64 v20; // rdx
  __int64 v21; // rsi
  __int64 v22; // r8
  __int64 v23; // r9
  __int64 v24; // r12
  unsigned __int64 v25; // rax
  int v26; // ecx
  unsigned __int64 *v27; // rdx
  __int64 v28; // rsi
  __int64 v29; // rbx
  __int64 v30; // r14
  __int64 v31; // rax
  signed __int64 v32; // r15
  __int64 *v33; // rax
  __int64 *v34; // rdi
  __int64 *v35; // rbx
  __int64 *v36; // r15
  _QWORD *v37; // r12
  __int16 *v38; // rax
  unsigned __int8 *v39; // rax
  char v40; // al
  __int64 v41; // rax
  __int64 v42; // rax
  unsigned __int64 v44; // rsi
  unsigned __int64 v45; // rbx
  __int64 v46; // [rsp-10h] [rbp-300h]
  __int64 v47; // [rsp-8h] [rbp-2F8h]
  __int64 v48; // [rsp+18h] [rbp-2D8h]
  __int64 *v49; // [rsp+30h] [rbp-2C0h]
  __int64 v50; // [rsp+38h] [rbp-2B8h]
  __int64 v51; // [rsp+50h] [rbp-2A0h]
  __int64 *v52; // [rsp+58h] [rbp-298h]
  __int64 *v53; // [rsp+68h] [rbp-288h]
  __int64 v54; // [rsp+80h] [rbp-270h]
  __int64 v56; // [rsp+90h] [rbp-260h]
  __int64 v57; // [rsp+90h] [rbp-260h]
  __int64 *v58; // [rsp+98h] [rbp-258h]
  __int64 v59; // [rsp+A8h] [rbp-248h] BYREF
  __int64 *v60; // [rsp+B0h] [rbp-240h] BYREF
  __int64 v61; // [rsp+B8h] [rbp-238h]
  _BYTE v62[32]; // [rsp+C0h] [rbp-230h] BYREF
  __int64 v63[8]; // [rsp+E0h] [rbp-210h] BYREF
  __int64 v64; // [rsp+120h] [rbp-1D0h] BYREF
  char *v65; // [rsp+128h] [rbp-1C8h]
  __int64 v66; // [rsp+130h] [rbp-1C0h]
  int v67; // [rsp+138h] [rbp-1B8h]
  char v68; // [rsp+13Ch] [rbp-1B4h]
  char v69; // [rsp+140h] [rbp-1B0h] BYREF
  __int64 v70; // [rsp+160h] [rbp-190h] BYREF
  __int16 *v71; // [rsp+168h] [rbp-188h]
  __int64 v72; // [rsp+170h] [rbp-180h]
  int v73; // [rsp+178h] [rbp-178h]
  char v74; // [rsp+17Ch] [rbp-174h]
  __int16 v75; // [rsp+180h] [rbp-170h] BYREF
  unsigned __int64 *v76; // [rsp+1A0h] [rbp-150h] BYREF
  __int64 v77; // [rsp+1A8h] [rbp-148h]
  _BYTE v78[32]; // [rsp+1B0h] [rbp-140h] BYREF
  __int64 v79; // [rsp+1D0h] [rbp-120h]
  __int64 v80; // [rsp+1D8h] [rbp-118h]
  __int16 v81; // [rsp+1E0h] [rbp-110h]
  __int64 v82; // [rsp+1E8h] [rbp-108h]
  void **v83; // [rsp+1F0h] [rbp-100h]
  void **v84; // [rsp+1F8h] [rbp-F8h]
  __int64 v85; // [rsp+200h] [rbp-F0h]
  int v86; // [rsp+208h] [rbp-E8h]
  __int16 v87; // [rsp+20Ch] [rbp-E4h]
  char v88; // [rsp+20Eh] [rbp-E2h]
  __int64 v89; // [rsp+210h] [rbp-E0h]
  __int64 v90; // [rsp+218h] [rbp-D8h]
  void *v91; // [rsp+220h] [rbp-D0h] BYREF
  void *v92; // [rsp+228h] [rbp-C8h] BYREF
  __int64 *v93; // [rsp+230h] [rbp-C0h] BYREF
  __int64 v94; // [rsp+238h] [rbp-B8h]
  _BYTE v95[176]; // [rsp+240h] [rbp-B0h] BYREF

  v4 = *a1;
  v64 = 0;
  v65 = &v69;
  v66 = 4;
  v67 = 0;
  v68 = 1;
  v82 = sub_BD5C60(v4);
  v83 = &v91;
  v84 = &v92;
  v76 = (unsigned __int64 *)v78;
  v91 = &unk_49DA100;
  v77 = 0x200000000LL;
  v87 = 512;
  v92 = &unk_49DA0B0;
  v85 = 0;
  v86 = 0;
  v88 = 7;
  v89 = 0;
  v90 = 0;
  v79 = 0;
  v80 = 0;
  v81 = 0;
  v49 = &a1[a2];
  if ( v49 != a1 )
  {
    v53 = a1;
    v5 = 0;
    while ( 1 )
    {
      v6 = *v53;
      if ( *(_DWORD *)(*v53 + 88) )
      {
        sub_11D2BF0((__int64)v63, 0);
        v7 = (unsigned __int8 *)sub_BD5D20(v6);
        sub_11D2C80(v63, *(_QWORD *)(v6 + 8), v7, v8);
        sub_11D33F0(v63, *(_QWORD *)(v6 + 40), v6);
        v51 = v6 - 32;
        sub_11D33F0(v63, *(_QWORD *)(v6 - 32LL * *(unsigned int *)(v6 + 88) - 64), v6);
        v11 = *(unsigned int *)(v6 + 88);
        v93 = (__int64 *)v95;
        v94 = 0x1000000000LL;
        if ( (_DWORD)v11 )
        {
          v12 = (__int64)v95;
          v13 = 1;
          v14 = 0;
          v15 = *(_QWORD *)(v6 - 32 * v11 - 32);
          while ( 1 )
          {
            *(_QWORD *)(v12 + 8 * v14) = v15;
            v14 = (unsigned int)(v94 + 1);
            LODWORD(v94) = v94 + 1;
            if ( v13 == v11 )
              break;
            v9 = HIDWORD(v94);
            v15 = *(_QWORD *)(v51 + 32 * (v13 - *(unsigned int *)(v6 + 88)));
            if ( v14 + 1 > (unsigned __int64)HIDWORD(v94) )
            {
              v57 = *(_QWORD *)(v51 + 32 * (v13 - *(unsigned int *)(v6 + 88)));
              sub_C8D5F0((__int64)&v93, v95, v14 + 1, 8u, v15, v10);
              v14 = (unsigned int)v94;
              v15 = v57;
            }
            v12 = (__int64)v93;
            ++v13;
          }
          v16 = v93;
          v52 = &v93[v14];
          if ( v52 != v93 )
          {
            v58 = v93;
            v50 = v6;
            while ( 1 )
            {
              v17 = *v58;
              if ( v68 )
              {
                v18 = v65;
                v9 = HIDWORD(v66);
                v12 = (__int64)&v65[8 * HIDWORD(v66)];
                if ( v65 != (char *)v12 )
                {
                  while ( v17 != *(_QWORD *)v18 )
                  {
                    v18 += 8;
                    if ( (char *)v12 == v18 )
                      goto LABEL_76;
                  }
                  goto LABEL_18;
                }
LABEL_76:
                if ( HIDWORD(v66) < (unsigned int)v66 )
                {
                  ++HIDWORD(v66);
                  *(_QWORD *)v12 = v17;
                  ++v64;
LABEL_24:
                  v19 = *(_QWORD *)(v17 + 56);
                  if ( !v19 )
                    BUG();
                  v20 = *(_QWORD *)(v19 + 16);
                  v80 = *(_QWORD *)(v17 + 56);
                  v81 = 0;
                  v79 = v20;
                  v21 = *(_QWORD *)sub_B46C60(v19 - 24);
                  v70 = v21;
                  if ( !v21 || (sub_B96E90((__int64)&v70, v21, 1), (v24 = v70) == 0) )
                  {
                    sub_93FB40((__int64)&v76, 0);
                    v24 = v70;
                    goto LABEL_82;
                  }
                  v25 = (unsigned __int64)v76;
                  v26 = v77;
                  v27 = &v76[2 * (unsigned int)v77];
                  if ( v76 == v27 )
                  {
LABEL_78:
                    if ( (unsigned int)v77 >= (unsigned __int64)HIDWORD(v77) )
                    {
                      v44 = (unsigned int)v77 + 1LL;
                      v45 = v48 & 0xFFFFFFFF00000000LL;
                      v48 &= 0xFFFFFFFF00000000LL;
                      if ( HIDWORD(v77) < v44 )
                      {
                        sub_C8D5F0((__int64)&v76, v78, v44, 0x10u, v22, v23);
                        v27 = &v76[2 * (unsigned int)v77];
                      }
                      *v27 = v45;
                      v27[1] = v24;
                      v24 = v70;
                      LODWORD(v77) = v77 + 1;
                    }
                    else
                    {
                      if ( v27 )
                      {
                        *(_DWORD *)v27 = 0;
                        v27[1] = v24;
                        v26 = v77;
                        v24 = v70;
                      }
                      LODWORD(v77) = v26 + 1;
                    }
LABEL_82:
                    if ( v24 )
LABEL_32:
                      sub_B91220((__int64)&v70, v24);
                    HIDWORD(v60) = 0;
                    v75 = 257;
                    v28 = *(_QWORD *)(v50 + 8);
                    v59 = v50;
                    v54 = sub_B35180((__int64)&v76, v28, 0x13u, (__int64)&v59, 1u, (unsigned int)v60, (__int64)&v70);
                    sub_11D33F0(v63, v17, v54);
                    v70 = 0;
                    v74 = 1;
                    v71 = &v75;
                    v72 = 4;
                    v73 = 0;
                    v29 = *(_QWORD *)(v50 + 16);
                    v30 = *(_QWORD *)(v51 - 32LL * *(unsigned int *)(v50 + 88) - 32);
                    v12 = v46;
                    v56 = *(_QWORD *)(v54 + 40);
                    v9 = v47;
                    v60 = (__int64 *)v62;
                    v61 = 0x400000000LL;
                    if ( v29 )
                    {
                      v31 = v29;
                      v32 = 0;
                      do
                      {
                        v31 = *(_QWORD *)(v31 + 8);
                        ++v32;
                      }
                      while ( v31 );
                      v33 = (__int64 *)v62;
                      if ( v32 > 4 )
                      {
                        sub_C8D5F0((__int64)&v60, v62, v32, 8u, v15, v10);
                        v9 = (unsigned int)v61;
                        v33 = &v60[(unsigned int)v61];
                      }
                      do
                      {
                        *v33 = v29;
                        v29 = *(_QWORD *)(v29 + 8);
                        ++v33;
                      }
                      while ( v29 );
                      v34 = v60;
                      v12 = (unsigned int)v61 + v32;
                      LODWORD(v61) = v61 + v32;
                      v35 = &v60[(unsigned int)v12];
                      if ( v35 != v60 )
                      {
                        v36 = v60;
                        while ( 2 )
                        {
                          while ( 2 )
                          {
                            v37 = (_QWORD *)*v36;
                            if ( v74 )
                            {
                              v38 = v71;
                              v12 = (__int64)&v71[4 * HIDWORD(v72)];
                              if ( v71 != (__int16 *)v12 )
                              {
                                while ( v37 != *(_QWORD **)v38 )
                                {
                                  v38 += 4;
                                  if ( (__int16 *)v12 == v38 )
                                    goto LABEL_60;
                                }
                                break;
                              }
LABEL_60:
                              if ( HIDWORD(v72) < (unsigned int)v72 )
                              {
                                ++HIDWORD(v72);
                                *(_QWORD *)v12 = v37;
                                ++v70;
                                goto LABEL_54;
                              }
                            }
                            sub_C8CC70((__int64)&v70, *v36, v12, v9, v15, v10);
                            if ( (_BYTE)v12 )
                            {
LABEL_54:
                              v39 = (unsigned __int8 *)v37[3];
                              v12 = *v39;
                              if ( (unsigned __int8)v12 > 0x1Cu )
                              {
                                if ( (_BYTE)v12 == 85 )
                                {
                                  v12 = *((_QWORD *)v39 - 4);
                                  if ( v12 )
                                  {
                                    if ( !*(_BYTE *)v12 )
                                    {
                                      v9 = *((_QWORD *)v39 + 10);
                                      if ( *(_QWORD *)(v12 + 24) == v9
                                        && (*(_BYTE *)(v12 + 33) & 0x20) != 0
                                        && *(_DWORD *)(v12 + 36) == 19 )
                                      {
                                        if ( v35 == ++v36 )
                                          goto LABEL_47;
                                        continue;
                                      }
                                    }
                                  }
                                }
                                v9 = v56;
                                if ( v56 == *((_QWORD *)v39 + 5) )
                                {
                                  if ( *v37 )
                                  {
                                    v12 = v37[2];
                                    v41 = v37[1];
                                    *(_QWORD *)v12 = v41;
                                    if ( v41 )
                                    {
                                      v12 = v37[2];
                                      *(_QWORD *)(v41 + 16) = v12;
                                    }
                                  }
                                  *v37 = v54;
                                  v42 = *(_QWORD *)(v54 + 16);
                                  v37[1] = v42;
                                  if ( v42 )
                                  {
                                    v12 = (__int64)(v37 + 1);
                                    *(_QWORD *)(v42 + 16) = v37 + 1;
                                  }
                                  ++v36;
                                  v37[2] = v54 + 16;
                                  *(_QWORD *)(v54 + 16) = v37;
                                  if ( v35 == v36 )
                                  {
LABEL_47:
                                    v34 = v60;
                                    goto LABEL_48;
                                  }
                                  continue;
                                }
                              }
                              sub_B19BE0(a3, v30, (__int64)v37);
                              if ( !v40 )
                              {
                                ++v36;
                                sub_11D9630(v63, (__int64)v37);
                                if ( v35 == v36 )
                                  goto LABEL_47;
                                continue;
                              }
                            }
                            break;
                          }
                          if ( v35 == ++v36 )
                            goto LABEL_47;
                          continue;
                        }
                      }
LABEL_48:
                      if ( v34 != (__int64 *)v62 )
                        _libc_free((unsigned __int64)v34);
                    }
                    if ( !v74 )
                      _libc_free((unsigned __int64)v71);
                    v5 = 1;
                    goto LABEL_18;
                  }
                  while ( *(_DWORD *)v25 )
                  {
                    v25 += 16LL;
                    if ( v27 == (unsigned __int64 *)v25 )
                      goto LABEL_78;
                  }
                  *(_QWORD *)(v25 + 8) = v70;
                  goto LABEL_32;
                }
              }
              sub_C8CC70((__int64)&v64, *v58, v12, v9, v15, v10);
              if ( (_BYTE)v12 )
                goto LABEL_24;
LABEL_18:
              if ( v52 == ++v58 )
              {
                v16 = v93;
                break;
              }
            }
          }
          if ( v16 != (__int64 *)v95 )
            _libc_free((unsigned __int64)v16);
        }
        sub_11D2C20(v63);
      }
      if ( v49 == ++v53 )
        goto LABEL_86;
    }
  }
  v5 = 0;
LABEL_86:
  nullsub_61();
  v91 = &unk_49DA100;
  nullsub_63();
  if ( v76 != (unsigned __int64 *)v78 )
    _libc_free((unsigned __int64)v76);
  if ( !v68 )
    _libc_free((unsigned __int64)v65);
  return v5;
}
