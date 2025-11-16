// Function: sub_27C40B0
// Address: 0x27c40b0
//
__int64 __fastcall sub_27C40B0(__int64 a1, unsigned __int64 a2)
{
  __int64 v2; // rdi
  __int64 v3; // rax
  __int64 v4; // rdx
  __int64 v5; // r9
  __int64 v6; // r12
  __int64 v7; // r14
  __int64 v8; // rdx
  __int64 *v9; // r15
  unsigned __int64 v10; // rcx
  int v11; // eax
  unsigned __int64 *v12; // rdi
  __int64 v13; // rax
  __int64 v14; // rax
  __int64 *v15; // r14
  unsigned int v16; // r12d
  __int64 *v17; // rbx
  __int64 v18; // r13
  __int64 v19; // rcx
  unsigned __int8 v20; // r15
  __int64 v21; // rsi
  _QWORD *v22; // rax
  _QWORD *v23; // rdx
  _BYTE **v24; // rax
  _BYTE *v25; // r15
  __int64 v26; // rax
  __int64 v27; // rdx
  __int64 v28; // rcx
  __int64 v29; // rdx
  __int64 v30; // rax
  __int64 v31; // rax
  __int64 v32; // rax
  __int64 v33; // rdx
  __int64 v34; // rcx
  __int64 v35; // rdx
  __int64 v36; // rcx
  __int64 v37; // rax
  unsigned __int64 v38; // rbx
  __int64 v39; // rax
  char *v41; // r15
  int v42; // r15d
  __int64 *v43; // rax
  unsigned int v44; // eax
  unsigned int v45; // edx
  _QWORD *v46; // rax
  unsigned __int8 *v47; // rax
  __int64 v48; // rdx
  __int64 v49; // rax
  unsigned __int8 *v50; // rax
  __int64 v51; // rdx
  __int64 v52; // rax
  __int64 v53; // r9
  unsigned __int8 *v54; // rax
  __int64 v55; // rdx
  __int64 v56; // rax
  int v57; // ecx
  int v58; // edx
  _QWORD *v59; // rdi
  __int64 *v60; // rax
  __int64 v61; // rax
  __int64 v62; // rax
  __int64 v63; // rax
  __int16 v64; // dx
  __int64 v65; // r8
  char v66; // cl
  char v67; // al
  unsigned __int16 v68; // dx
  _QWORD *v69; // rax
  __int64 v70; // r15
  unsigned int v71; // eax
  int v72; // edi
  unsigned int v73; // r8d
  unsigned int v74; // eax
  __int64 v75; // [rsp+8h] [rbp-1E8h]
  unsigned __int8 v76; // [rsp+1Fh] [rbp-1D1h]
  __int64 v77; // [rsp+20h] [rbp-1D0h]
  __int64 v78; // [rsp+20h] [rbp-1D0h]
  __int64 v79; // [rsp+30h] [rbp-1C0h]
  __int64 v80; // [rsp+30h] [rbp-1C0h]
  __int64 v81; // [rsp+38h] [rbp-1B8h]
  _QWORD *v82; // [rsp+38h] [rbp-1B8h]
  __int64 v83; // [rsp+40h] [rbp-1B0h]
  __int64 v84; // [rsp+48h] [rbp-1A8h]
  __int64 v85; // [rsp+48h] [rbp-1A8h]
  int v86; // [rsp+50h] [rbp-1A0h]
  unsigned __int16 v87; // [rsp+50h] [rbp-1A0h]
  _BOOL8 v88; // [rsp+58h] [rbp-198h]
  int v89; // [rsp+58h] [rbp-198h]
  __int64 v90; // [rsp+58h] [rbp-198h]
  __int64 v91; // [rsp+58h] [rbp-198h]
  __int64 v92; // [rsp+58h] [rbp-198h]
  __int64 v93; // [rsp+60h] [rbp-190h]
  unsigned __int64 *v94; // [rsp+60h] [rbp-190h]
  unsigned __int64 *v95; // [rsp+60h] [rbp-190h]
  __int64 v98; // [rsp+80h] [rbp-170h] BYREF
  __int64 v99; // [rsp+88h] [rbp-168h] BYREF
  __int64 v100; // [rsp+90h] [rbp-160h] BYREF
  __int64 v101; // [rsp+98h] [rbp-158h]
  unsigned __int64 v102[2]; // [rsp+A0h] [rbp-150h] BYREF
  __int64 v103; // [rsp+B0h] [rbp-140h]
  __int64 v104; // [rsp+C0h] [rbp-130h] BYREF
  __int64 v105; // [rsp+C8h] [rbp-128h]
  char *v106; // [rsp+D0h] [rbp-120h]
  __int16 v107; // [rsp+E0h] [rbp-110h]
  __int64 *v108; // [rsp+F0h] [rbp-100h] BYREF
  __int64 v109; // [rsp+F8h] [rbp-F8h]
  _BYTE v110[240]; // [rsp+100h] [rbp-F0h] BYREF

  v2 = **(_QWORD **)(a2 + 32);
  v108 = (__int64 *)v110;
  v109 = 0x800000000LL;
  v3 = sub_AA5930(v2);
  if ( v3 != v4 )
  {
    v6 = v4;
    v7 = v3;
    do
    {
      v106 = (char *)v7;
      v104 = 6;
      v105 = 0;
      if ( v7 != -4096 && v7 != 0 && v7 != -8192 )
        sub_BD73F0((__int64)&v104);
      v8 = (unsigned int)v109;
      v9 = &v104;
      v10 = (unsigned __int64)v108;
      v11 = v109;
      if ( (unsigned __int64)(unsigned int)v109 + 1 > HIDWORD(v109) )
      {
        if ( v108 > &v104 || &v104 >= &v108[3 * (unsigned int)v109] )
        {
          v95 = (unsigned __int64 *)sub_C8D7D0((__int64)&v108, (__int64)v110, (unsigned int)v109 + 1LL, 0x18u, v102, v5);
          sub_F17F80((__int64)&v108, v95);
          v42 = v102[0];
          v10 = (unsigned __int64)v95;
          if ( v108 != (__int64 *)v110 )
          {
            _libc_free((unsigned __int64)v108);
            v10 = (unsigned __int64)v95;
          }
          v8 = (unsigned int)v109;
          HIDWORD(v109) = v42;
          v9 = &v104;
          v108 = (__int64 *)v10;
          v11 = v109;
        }
        else
        {
          v41 = (char *)((char *)&v104 - (char *)v108);
          v94 = (unsigned __int64 *)sub_C8D7D0((__int64)&v108, (__int64)v110, (unsigned int)v109 + 1LL, 0x18u, v102, v5);
          sub_F17F80((__int64)&v108, v94);
          v10 = (unsigned __int64)v94;
          if ( v108 == (__int64 *)v110 )
          {
            v108 = (__int64 *)v94;
            HIDWORD(v109) = v102[0];
          }
          else
          {
            v89 = v102[0];
            _libc_free((unsigned __int64)v108);
            v10 = (unsigned __int64)v94;
            v108 = (__int64 *)v94;
            HIDWORD(v109) = v89;
          }
          v8 = (unsigned int)v109;
          v9 = (__int64 *)&v41[v10];
          v11 = v109;
        }
      }
      v12 = (unsigned __int64 *)(v10 + 24 * v8);
      if ( v12 )
      {
        *v12 = 6;
        v13 = v9[2];
        v12[1] = 0;
        v12[2] = v13;
        if ( v13 != 0 && v13 != -4096 && v13 != -8192 )
          sub_BD6050(v12, *v9 & 0xFFFFFFFFFFFFFFF8LL);
        v11 = v109;
      }
      LODWORD(v109) = v11 + 1;
      if ( v106 != 0 && v106 + 4096 != 0 && v106 != (char *)-8192LL )
        sub_BD60C0(&v104);
      if ( !v7 )
        BUG();
      v14 = *(_QWORD *)(v7 + 32);
      if ( !v14 )
        BUG();
      v7 = 0;
      if ( *(_BYTE *)(v14 - 24) == 84 )
        v7 = v14 - 24;
    }
    while ( v6 != v7 );
  }
  v15 = v108;
  v16 = 0;
  if ( &v108[3 * (unsigned int)v109] != v108 )
  {
    v17 = &v108[3 * (unsigned int)v109];
    v93 = a2 + 56;
    do
    {
      v18 = v15[2];
      if ( v18 && *(_BYTE *)v18 == 84 )
      {
        v19 = *(_QWORD *)(v18 - 8);
        v20 = *(_BYTE *)(a2 + 84);
        v21 = *(_QWORD *)(v19 + 32LL * *(unsigned int *)(v18 + 72));
        if ( v20 )
        {
          v22 = *(_QWORD **)(a2 + 64);
          v23 = &v22[*(unsigned int *)(a2 + 76)];
          if ( v22 == v23 )
          {
LABEL_69:
            v88 = 0;
          }
          else
          {
            while ( v21 != *v22 )
            {
              if ( v23 == ++v22 )
                goto LABEL_69;
            }
            v88 = 1;
            v19 += 32;
            v20 = 0;
          }
        }
        else
        {
          v43 = sub_C8CA60(v93, v21);
          v20 = v43 == 0;
          v88 = v43 != 0;
          v19 = *(_QWORD *)(v18 - 8) + 32 * v88;
        }
        if ( **(_BYTE **)v19 == 18 )
        {
          if ( (unsigned __int8)sub_27BFC70((void **)(*(_QWORD *)v19 + 24LL), &v98) )
          {
            v81 = v20;
            v24 = (_BYTE **)(*(_QWORD *)(v18 - 8) + 32LL * v20);
            v25 = *v24;
            if ( **v24 == 43 )
            {
              v26 = *((_QWORD *)v25 - 4);
              if ( *(_BYTE *)v26 == 18 )
              {
                v27 = *((_QWORD *)v25 - 8);
                if ( v27 )
                {
                  if ( v18 == v27 )
                  {
                    if ( (unsigned __int8)sub_27BFC70((void **)(v26 + 24), &v99) )
                    {
                      v29 = *((_QWORD *)v25 + 2);
                      v30 = *(_QWORD *)(v29 + 8);
                      if ( v30 )
                      {
                        if ( !*(_QWORD *)(v30 + 8) )
                        {
                          v84 = *(_QWORD *)(v29 + 24);
                          if ( *(_BYTE *)v84 == 83 || (v84 = *(_QWORD *)(v30 + 24), *(_BYTE *)v84 == 83) )
                          {
                            v31 = *(_QWORD *)(v84 + 16);
                            if ( v31 )
                            {
                              if ( !*(_QWORD *)(v31 + 8) )
                              {
                                v32 = *(_QWORD *)(v31 + 24);
                                v77 = v32;
                                if ( *(_BYTE *)v32 == 31
                                  && (unsigned __int8)sub_B19060(v93, *(_QWORD *)(v32 + 40), v29, v28)
                                  && (!(unsigned __int8)sub_B19060(v93, *(_QWORD *)(v77 - 32), v33, v34)
                                   || !(unsigned __int8)sub_B19060(v93, *(_QWORD *)(v77 - 64), v35, v36)) )
                                {
                                  v37 = *(_QWORD *)(v84 - 32);
                                  if ( *(_BYTE *)v37 == 18 )
                                  {
                                    v76 = sub_27BFC70((void **)(v37 + 24), &v100);
                                    if ( v76 )
                                    {
                                      switch ( *(_WORD *)(v84 + 2) & 0x3F )
                                      {
                                        case 1:
                                        case 9:
                                          v86 = 32;
                                          goto LABEL_75;
                                        case 2:
                                        case 0xA:
                                          v86 = 38;
                                          goto LABEL_75;
                                        case 3:
                                        case 0xB:
                                          v86 = 39;
                                          goto LABEL_75;
                                        case 4:
                                        case 0xC:
                                          v86 = 40;
                                          goto LABEL_75;
                                        case 5:
                                        case 0xD:
                                          v86 = 41;
                                          goto LABEL_75;
                                        case 6:
                                        case 0xE:
                                          v86 = 33;
LABEL_75:
                                          if ( v98 != (int)v98 || v99 != (int)v99 || (int)v100 != v100 || !v99 )
                                            break;
                                          if ( v99 <= 0 )
                                          {
                                            if ( v98 <= v100 )
                                              break;
                                            v71 = v98 - v100;
                                            if ( (unsigned int)(v86 - 39) <= 1 )
                                            {
                                              v74 = v71 + 1;
                                              if ( !v74 )
                                                break;
                                              v72 = v99;
                                              v73 = v74 % -(int)v99;
                                            }
                                            else
                                            {
                                              v72 = v99;
                                              v45 = v71 % -(int)v99;
                                              v73 = v45;
                                              if ( (unsigned int)(v86 - 32) <= 1 )
                                                goto LABEL_121;
                                            }
                                            if ( v73 && v100 < v72 + (int)v100 )
                                              break;
                                          }
                                          else
                                          {
                                            if ( v98 >= v100 )
                                              break;
                                            v44 = v100 - v98;
                                            if ( (v86 == 41 || v86 == 38) && !++v44 )
                                              break;
                                            v45 = v44 % (unsigned int)v99;
                                            if ( (unsigned int)(v86 - 32) <= 1 )
                                            {
LABEL_121:
                                              if ( v45 )
                                                break;
                                              goto LABEL_86;
                                            }
                                            if ( v45 && v100 > (int)v99 + (int)v100 )
                                              break;
                                          }
LABEL_86:
                                          v46 = (_QWORD *)sub_BD5C60(v18);
                                          v79 = sub_BCB2D0(v46);
                                          v47 = (unsigned __int8 *)sub_BD5D20(v18);
                                          v107 = 773;
                                          v104 = (__int64)v47;
                                          v105 = v48;
                                          v106 = ".int";
                                          v83 = sub_BD2DA0(80);
                                          if ( v83 )
                                          {
                                            sub_B44260(v83, v79, 55, 0x8000000u, v18 + 24, 0);
                                            *(_DWORD *)(v83 + 72) = 2;
                                            sub_BD6B50((unsigned __int8 *)v83, (const char **)&v104);
                                            sub_BD2A10(v83, *(_DWORD *)(v83 + 72), 1);
                                          }
                                          v90 = *(_QWORD *)(*(_QWORD *)(v18 - 8)
                                                          + 32LL * *(unsigned int *)(v18 + 72)
                                                          + 8 * v88);
                                          v49 = sub_ACD640(v79, v98, 1u);
                                          sub_F0A850(v83, v49, v90);
                                          v104 = *(_QWORD *)(v18 + 48);
                                          if ( v104 )
                                            sub_27C0030(&v104);
                                          if ( (__int64 *)(v83 + 48) != &v104 )
                                            sub_27C0330((__int64 *)(v83 + 48), (unsigned __int8 **)&v104);
                                          sub_9C6650(&v104);
                                          v50 = (unsigned __int8 *)sub_BD5D20((__int64)v25);
                                          v107 = 773;
                                          v104 = (__int64)v50;
                                          v105 = v51;
                                          v106 = ".int";
                                          v52 = sub_ACD640(v79, v99, 1u);
                                          v53 = v75;
                                          LOWORD(v53) = 0;
                                          v91 = sub_B504D0(13, v83, v52, (__int64)&v104, (__int64)(v25 + 24), v53);
                                          v104 = *((_QWORD *)v25 + 6);
                                          if ( v104 )
                                            sub_27C0030(&v104);
                                          if ( (__int64 *)(v91 + 48) != &v104 )
                                            sub_27C0330((__int64 *)(v91 + 48), (unsigned __int8 **)&v104);
                                          sub_9C6650(&v104);
                                          sub_F0A850(
                                            v83,
                                            v91,
                                            *(_QWORD *)(*(_QWORD *)(v18 - 8)
                                                      + 32LL * *(unsigned int *)(v18 + 72)
                                                      + 8 * v81));
                                          v78 = v77 + 24;
                                          v80 = sub_ACD640(v79, v100, 1u);
                                          v54 = (unsigned __int8 *)sub_BD5D20(v84);
                                          v107 = 261;
                                          v104 = (__int64)v54;
                                          v105 = v55;
                                          v82 = sub_BD2C40(72, unk_3F10FD0);
                                          if ( v82 )
                                          {
                                            v56 = *(_QWORD *)(v91 + 8);
                                            v57 = *(unsigned __int8 *)(v56 + 8);
                                            if ( (unsigned int)(v57 - 17) > 1 )
                                            {
                                              v61 = sub_BCB2A0(*(_QWORD **)v56);
                                            }
                                            else
                                            {
                                              v58 = *(_DWORD *)(v56 + 32);
                                              v59 = *(_QWORD **)v56;
                                              BYTE4(v101) = (_BYTE)v57 == 18;
                                              LODWORD(v101) = v58;
                                              v60 = (__int64 *)sub_BCB2A0(v59);
                                              v61 = sub_BCE1B0(v60, v101);
                                            }
                                            sub_B523C0((__int64)v82, v61, 53, v86, v91, v80, (__int64)&v104, v78, 0, 0);
                                          }
                                          v104 = *(_QWORD *)(v84 + 48);
                                          if ( v104 )
                                            sub_27C0030(&v104);
                                          if ( v82 + 6 != &v104 )
                                            sub_27C0330(v82 + 6, (unsigned __int8 **)&v104);
                                          sub_9C6650(&v104);
                                          v103 = v18;
                                          v102[0] = 6;
                                          v102[1] = 0;
                                          if ( v18 != -4096 && v18 != -8192 )
                                            sub_BD73F0((__int64)v102);
                                          sub_BD6B90((unsigned __int8 *)v82, (unsigned __int8 *)v84);
                                          sub_BD84D0(v84, (__int64)v82);
                                          v106 = 0;
                                          sub_F5CAB0(
                                            (char *)v84,
                                            *(__int64 **)(a1 + 32),
                                            *(_QWORD **)(a1 + 48),
                                            (__int64)&v104);
                                          sub_A17130((__int64)&v104);
                                          v62 = sub_ACADE0(*((__int64 ***)v25 + 1));
                                          sub_BD84D0((__int64)v25, v62);
                                          v106 = 0;
                                          sub_F5CAB0(v25, *(__int64 **)(a1 + 32), *(_QWORD **)(a1 + 48), (__int64)&v104);
                                          sub_A17130((__int64)&v104);
                                          if ( v103 )
                                          {
                                            v63 = *(_QWORD *)(v18 + 8);
                                            v107 = 259;
                                            v92 = v63;
                                            v104 = (__int64)"indvar.conv";
                                            v65 = sub_AA5190(*(_QWORD *)(v18 + 40));
                                            if ( v65 )
                                            {
                                              v66 = v64;
                                              v67 = HIBYTE(v64);
                                            }
                                            else
                                            {
                                              v67 = 0;
                                              v66 = 0;
                                            }
                                            v85 = v65;
                                            LOBYTE(v68) = v66;
                                            HIBYTE(v68) = v67;
                                            v87 = v68;
                                            v69 = sub_BD2C40(72, unk_3F10A14);
                                            v70 = (__int64)v69;
                                            if ( v69 )
                                              sub_B518D0((__int64)v69, v83, v92, (__int64)&v104, v85, v87);
                                            v104 = *(_QWORD *)(v18 + 48);
                                            if ( v104 )
                                              sub_27C0030(&v104);
                                            if ( (__int64 *)(v70 + 48) != &v104 )
                                              sub_27C0330((__int64 *)(v70 + 48), (unsigned __int8 **)&v104);
                                            sub_9C6650(&v104);
                                            sub_BD84D0(v18, v70);
                                            v106 = 0;
                                            sub_F5CAB0(
                                              (char *)v18,
                                              *(__int64 **)(a1 + 32),
                                              *(_QWORD **)(a1 + 48),
                                              (__int64)&v104);
                                            sub_A17130((__int64)&v104);
                                            if ( v103 != -4096 && v103 != 0 && v103 != -8192 )
                                              sub_BD60C0(v102);
                                          }
                                          v16 = v76;
                                          break;
                                        default:
                                          break;
                                      }
                                    }
                                  }
                                }
                              }
                            }
                          }
                        }
                      }
                    }
                  }
                }
              }
            }
          }
        }
      }
      v15 += 3;
    }
    while ( v15 != v17 );
    if ( (_BYTE)v16 )
      sub_DAC210(*(_QWORD *)(a1 + 8), a2);
    v38 = (unsigned __int64)v108;
    v15 = &v108[3 * (unsigned int)v109];
    if ( v108 != v15 )
    {
      do
      {
        v39 = *(v15 - 1);
        v15 -= 3;
        if ( v39 != -4096 && v39 != 0 && v39 != -8192 )
          sub_BD60C0(v15);
      }
      while ( (__int64 *)v38 != v15 );
      v15 = v108;
    }
  }
  if ( v15 != (__int64 *)v110 )
    _libc_free((unsigned __int64)v15);
  return v16;
}
