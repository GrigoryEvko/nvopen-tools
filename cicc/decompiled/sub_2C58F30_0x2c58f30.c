// Function: sub_2C58F30
// Address: 0x2c58f30
//
__int64 __fastcall sub_2C58F30(__int64 a1, __int64 a2)
{
  unsigned __int8 *v3; // r13
  unsigned __int8 *v4; // r14
  int v5; // eax
  unsigned __int64 v6; // r10
  unsigned int v7; // r8d
  char *v8; // r11
  char v9; // dl
  __int64 v10; // r15
  __int64 v13; // rax
  unsigned int v14; // ecx
  unsigned int v15; // edi
  char v16; // al
  unsigned __int64 v17; // r10
  __int64 v18; // r11
  __int64 *v19; // rdi
  __int64 v20; // rax
  int v21; // edx
  __int64 v22; // rax
  __int64 v23; // r10
  __int64 v24; // rsi
  int v25; // edx
  int v26; // ecx
  unsigned __int64 v27; // rax
  __int64 v28; // rax
  int v29; // edx
  bool v30; // zf
  int v31; // edx
  __int64 v32; // rax
  int v33; // edx
  int v34; // r15d
  __int64 v35; // rax
  __int64 v36; // rcx
  int v37; // edx
  signed __int64 v38; // rax
  __int64 v39; // rdx
  bool v40; // of
  __int64 v41; // rdx
  __int64 v42; // rdi
  char *v43; // r11
  _BYTE *v44; // r15
  __int64 (__fastcall *v45)(__int64, _BYTE *, _BYTE *, __int64, __int64, __int64); // rax
  __int64 v46; // rax
  unsigned __int64 v47; // r10
  unsigned __int8 *v48; // rax
  char *v49; // r10
  __int64 v50; // r15
  __int64 v51; // r13
  __int64 i; // rbx
  _BOOL4 v53; // eax
  _BOOL4 v54; // eax
  _QWORD *v55; // rax
  _QWORD *v56; // r10
  void *v57; // rcx
  __int64 v58; // rax
  __int64 v59; // rcx
  __int64 v60; // r13
  __int64 v61; // rbx
  __int64 v62; // r15
  __int64 v63; // rdx
  unsigned int v64; // esi
  __int64 v65; // rax
  bool v66; // cc
  unsigned __int64 v67; // rax
  int v68; // [rsp+10h] [rbp-148h]
  __int64 v69; // [rsp+18h] [rbp-140h]
  __int64 v70; // [rsp+20h] [rbp-138h]
  __int64 v71; // [rsp+28h] [rbp-130h]
  __int64 v72; // [rsp+30h] [rbp-128h]
  unsigned __int64 v73; // [rsp+30h] [rbp-128h]
  signed __int64 v74; // [rsp+30h] [rbp-128h]
  int v75; // [rsp+40h] [rbp-118h]
  __int64 v76; // [rsp+40h] [rbp-118h]
  __int64 v77; // [rsp+40h] [rbp-118h]
  char *v78; // [rsp+48h] [rbp-110h]
  __int64 v79; // [rsp+48h] [rbp-110h]
  _DWORD *v80; // [rsp+48h] [rbp-110h]
  char *v81; // [rsp+48h] [rbp-110h]
  _QWORD *v82; // [rsp+48h] [rbp-110h]
  unsigned __int64 v83; // [rsp+48h] [rbp-110h]
  unsigned __int8 *v84; // [rsp+48h] [rbp-110h]
  __int64 v85; // [rsp+50h] [rbp-108h]
  __int64 v86; // [rsp+58h] [rbp-100h]
  __int64 v87; // [rsp+58h] [rbp-100h]
  char *v88; // [rsp+58h] [rbp-100h]
  char *v89; // [rsp+58h] [rbp-100h]
  char *v90; // [rsp+58h] [rbp-100h]
  char *v91; // [rsp+58h] [rbp-100h]
  __int64 v92; // [rsp+58h] [rbp-100h]
  char *v93; // [rsp+58h] [rbp-100h]
  unsigned __int64 v94; // [rsp+60h] [rbp-F8h]
  int v95; // [rsp+60h] [rbp-F8h]
  int v96; // [rsp+60h] [rbp-F8h]
  __int64 *v97; // [rsp+60h] [rbp-F8h]
  unsigned __int64 v98; // [rsp+60h] [rbp-F8h]
  unsigned __int64 v99; // [rsp+60h] [rbp-F8h]
  unsigned __int64 v100; // [rsp+60h] [rbp-F8h]
  unsigned int v101; // [rsp+68h] [rbp-F0h]
  unsigned __int8 v102; // [rsp+68h] [rbp-F0h]
  char v103; // [rsp+68h] [rbp-F0h]
  int v104[8]; // [rsp+70h] [rbp-E8h] BYREF
  __int16 v105; // [rsp+90h] [rbp-C8h]
  _BYTE v106[32]; // [rsp+A0h] [rbp-B8h] BYREF
  __int16 v107; // [rsp+C0h] [rbp-98h]
  _DWORD *v108; // [rsp+D0h] [rbp-88h] BYREF
  __int64 v109; // [rsp+D8h] [rbp-80h]
  _BYTE v110[120]; // [rsp+E0h] [rbp-78h] BYREF

  if ( *(_BYTE *)a2 == 92 )
  {
    v3 = *(unsigned __int8 **)(a2 - 64);
    if ( v3 && (v4 = *(unsigned __int8 **)(a2 - 32)) != 0 )
    {
      v5 = *v3;
      v6 = *(unsigned int *)(a2 + 80);
      v7 = 0;
      v8 = *(char **)(a2 + 72);
      if ( (unsigned __int8)(v5 - 67) <= 0xCu )
      {
        v9 = *v4;
        if ( (unsigned __int8)(*v4 - 67) <= 0xCu )
        {
          v10 = *(_QWORD *)(*((_QWORD *)v3 - 4) + 8LL);
          if ( v10 == *(_QWORD *)(*((_QWORD *)v4 - 4) + 8LL) )
          {
            if ( (_BYTE)v5 == v9 )
            {
              v101 = v5 - 29;
            }
            else
            {
              if ( (_BYTE)v5 != 69 )
              {
                v90 = *(char **)(a2 + 72);
                v98 = *(unsigned int *)(a2 + 80);
                v103 = *v4;
                if ( (_BYTE)v5 != 68 )
                  return v7;
                v53 = sub_B44910((__int64)v3);
                v9 = v103;
                v6 = v98;
                v8 = v90;
                v7 = v53;
                if ( !v53 )
                  return v7;
              }
              v101 = 40;
              if ( v9 != 69 )
              {
                v91 = v8;
                v7 = 0;
                v99 = v6;
                if ( v9 != 68 )
                  return v7;
                v54 = sub_B44910((__int64)v4);
                v6 = v99;
                v8 = v91;
                v7 = v54;
                if ( !v54 )
                  return v7;
              }
            }
            v7 = 0;
            v85 = *(_QWORD *)(a2 + 8);
            if ( *(_BYTE *)(v85 + 8) == 17 )
            {
              v13 = *((_QWORD *)v3 + 1);
              v86 = v13;
              if ( *(_BYTE *)(v13 + 8) == 17 && *(_BYTE *)(v10 + 8) == 17 )
              {
                v14 = *(_DWORD *)(v10 + 32);
                v15 = *(_DWORD *)(v13 + 32);
                if ( v15 == v14 )
                {
                  v108 = v110;
                  v109 = 0x1000000000LL;
                }
                else
                {
                  if ( v14 % v15 && v15 % v14 )
                    return v7;
                  v108 = v110;
                  v109 = 0x1000000000LL;
                  if ( v15 > v14 )
                  {
                    v78 = v8;
                    v94 = v6;
                    v16 = sub_9B8470(v15 / v14, v8, v6, (__int64)&v108);
                    v17 = v94;
                    v18 = (__int64)v78;
                    if ( !v16 )
                      goto LABEL_66;
LABEL_21:
                    v19 = (__int64 *)v10;
                    if ( (unsigned int)*(unsigned __int8 *)(v10 + 8) - 17 <= 1 )
                      v19 = **(__int64 ***)(v10 + 16);
                    v70 = v18;
                    v72 = v17;
                    v71 = sub_BCDA70(v19, v109);
                    v20 = sub_DFD060(*(__int64 **)(a1 + 152), (unsigned int)*v3 - 29, v86, v10);
                    v75 = v21;
                    v79 = v20;
                    v22 = sub_DFD060(*(__int64 **)(a1 + 152), (unsigned int)*v4 - 29, v86, v10);
                    v23 = v72;
                    v24 = v22;
                    v69 = v22;
                    v68 = v25;
                    v26 = 1;
                    if ( v25 != 1 )
                      v26 = v75;
                    v27 = v22 + v79;
                    v95 = v26;
                    if ( __OFADD__(v24, v79) )
                    {
                      v27 = 0x8000000000000000LL;
                      if ( v69 > 0 )
                        v27 = 0x7FFFFFFFFFFFFFFFLL;
                    }
                    v73 = v27;
                    v28 = sub_DFBC30(
                            *(__int64 **)(a1 + 152),
                            6,
                            v86,
                            v70,
                            v23,
                            *(unsigned int *)(a1 + 192),
                            0,
                            0,
                            0,
                            0,
                            a2);
                    v30 = v29 == 1;
                    v31 = 1;
                    if ( !v30 )
                      v31 = v95;
                    v96 = v31;
                    if ( __OFADD__(v28, v73) )
                    {
                      v66 = v28 <= 0;
                      v67 = 0x8000000000000000LL;
                      if ( !v66 )
                        v67 = 0x7FFFFFFFFFFFFFFFLL;
                      v74 = v67;
                    }
                    else
                    {
                      v74 = v28 + v73;
                    }
                    v32 = sub_DFBC30(
                            *(__int64 **)(a1 + 152),
                            6,
                            v10,
                            (__int64)v108,
                            (unsigned int)v109,
                            *(unsigned int *)(a1 + 192),
                            0,
                            0,
                            0,
                            0,
                            0);
                    v34 = v33;
                    v87 = v32;
                    v35 = sub_DFD060(*(__int64 **)(a1 + 152), v101, v85, v71);
                    v36 = v35;
                    if ( v37 == 1 )
                      v34 = 1;
                    v38 = v35 + v87;
                    if ( __OFADD__(v36, v87) )
                    {
                      v38 = 0x7FFFFFFFFFFFFFFFLL;
                      if ( v36 <= 0 )
                        v38 = 0x8000000000000000LL;
                    }
                    v39 = *((_QWORD *)v3 + 2);
                    if ( !v39 || *(_QWORD *)(v39 + 8) )
                    {
                      if ( v75 == 1 )
                        v34 = 1;
                      v40 = __OFADD__(v79, v38);
                      v38 += v79;
                      if ( v40 )
                      {
                        v38 = 0x7FFFFFFFFFFFFFFFLL;
                        if ( v79 <= 0 )
                          v38 = 0x8000000000000000LL;
                      }
                    }
                    v41 = *((_QWORD *)v4 + 2);
                    if ( !v41 || *(_QWORD *)(v41 + 8) )
                    {
                      if ( v68 == 1 )
                        v34 = 1;
                      v40 = __OFADD__(v69, v38);
                      v38 += v69;
                      if ( v40 )
                      {
                        if ( v69 <= 0 )
                        {
                          if ( v96 == v34 )
                            goto LABEL_43;
                          goto LABEL_65;
                        }
                        v38 = 0x7FFFFFFFFFFFFFFFLL;
                      }
                    }
                    if ( v96 == v34 )
                    {
                      if ( v74 >= v38 )
                        goto LABEL_43;
LABEL_66:
                      v7 = 0;
LABEL_59:
                      if ( v108 != (_DWORD *)v110 )
                      {
                        v102 = v7;
                        _libc_free((unsigned __int64)v108);
                        return v102;
                      }
                      return v7;
                    }
LABEL_65:
                    if ( v96 < v34 )
                      goto LABEL_66;
LABEL_43:
                    v42 = *(_QWORD *)(a1 + 88);
                    v105 = 257;
                    v43 = (char *)*((_QWORD *)v4 - 4);
                    v80 = v108;
                    v44 = (_BYTE *)*((_QWORD *)v3 - 4);
                    v97 = (__int64 *)(a1 + 8);
                    v76 = (unsigned int)v109;
                    v45 = *(__int64 (__fastcall **)(__int64, _BYTE *, _BYTE *, __int64, __int64, __int64))(*(_QWORD *)v42 + 112LL);
                    if ( v45 == sub_9B6630 )
                    {
                      if ( *v44 > 0x15u || (unsigned __int8)*v43 > 0x15u )
                        goto LABEL_77;
                      v88 = (char *)*((_QWORD *)v4 - 4);
                      v46 = sub_AD5CE0((__int64)v44, (__int64)v43, v108, (unsigned int)v109, 0);
                      v43 = v88;
                      v47 = v46;
                    }
                    else
                    {
                      v93 = (char *)*((_QWORD *)v4 - 4);
                      v65 = ((__int64 (__fastcall *)(__int64, _BYTE *, char *, _DWORD *, _QWORD))v45)(
                              v42,
                              v44,
                              v43,
                              v108,
                              (unsigned int)v109);
                      v43 = v93;
                      v47 = v65;
                    }
                    if ( v47 )
                    {
LABEL_48:
                      v107 = 257;
                      v89 = (char *)v47;
                      v48 = (unsigned __int8 *)sub_2C511B0(
                                                 v97,
                                                 v101,
                                                 v47,
                                                 (__int64 **)v85,
                                                 (__int64)v106,
                                                 0,
                                                 v104[0],
                                                 0);
                      v49 = v89;
                      v50 = (__int64)v48;
                      if ( *v48 > 0x1Cu )
                      {
                        sub_B45260(v48, (__int64)v3, 1);
                        sub_B45560((unsigned __int8 *)v50, (unsigned __int64)v4);
                        v49 = v89;
                      }
                      v51 = a1 + 200;
                      if ( (unsigned __int8)*v49 > 0x1Cu )
                        sub_F15FC0(a1 + 200, (__int64)v49);
                      sub_BD84D0(a2, v50);
                      if ( *(_BYTE *)v50 > 0x1Cu )
                      {
                        sub_BD6B90((unsigned __int8 *)v50, (unsigned __int8 *)a2);
                        for ( i = *(_QWORD *)(v50 + 16); i; i = *(_QWORD *)(i + 8) )
                          sub_F15FC0(v51, *(_QWORD *)(i + 24));
                        if ( *(_BYTE *)v50 > 0x1Cu )
                          sub_F15FC0(v51, v50);
                      }
                      v7 = 1;
                      if ( *(_BYTE *)a2 > 0x1Cu )
                      {
                        sub_F15FC0(v51, a2);
                        v7 = 1;
                      }
                      goto LABEL_59;
                    }
LABEL_77:
                    v92 = (__int64)v43;
                    v107 = 257;
                    v55 = sub_BD2C40(112, unk_3F1FE60);
                    v56 = v55;
                    if ( v55 )
                    {
                      v57 = v80;
                      v82 = v55;
                      sub_B4E9E0((__int64)v55, (__int64)v44, v92, v57, v76, (__int64)v106, 0, 0);
                      v56 = v82;
                    }
                    v83 = (unsigned __int64)v56;
                    (*(void (__fastcall **)(_QWORD, _QWORD *, int *, _QWORD, _QWORD))(**(_QWORD **)(a1 + 96) + 16LL))(
                      *(_QWORD *)(a1 + 96),
                      v56,
                      v104,
                      *(_QWORD *)(a1 + 64),
                      *(_QWORD *)(a1 + 72));
                    v58 = *(_QWORD *)(a1 + 8);
                    v47 = v83;
                    v59 = v58 + 16LL * *(unsigned int *)(a1 + 16);
                    if ( v58 != v59 )
                    {
                      v84 = v3;
                      v60 = v47;
                      v77 = a1;
                      v61 = *(_QWORD *)(a1 + 8);
                      v62 = v59;
                      do
                      {
                        v63 = *(_QWORD *)(v61 + 8);
                        v64 = *(_DWORD *)v61;
                        v61 += 16;
                        sub_B99FD0(v60, v64, v63);
                      }
                      while ( v62 != v61 );
                      v47 = v60;
                      a1 = v77;
                      v3 = v84;
                    }
                    goto LABEL_48;
                  }
                }
                v81 = v8;
                v100 = v6;
                sub_9B8300(v14 / v15, (unsigned int *)v8, v6, (__int64)&v108);
                v17 = v100;
                v18 = (__int64)v81;
                goto LABEL_21;
              }
            }
          }
        }
      }
    }
    else
    {
      return 0;
    }
    return v7;
  }
  return 0;
}
