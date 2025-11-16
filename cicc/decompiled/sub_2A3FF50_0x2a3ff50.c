// Function: sub_2A3FF50
// Address: 0x2a3ff50
//
__int64 __fastcall sub_2A3FF50(__int64 a1, __int64 *a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  __int64 v6; // r12
  __int64 *v7; // rbx
  __int64 v8; // r12
  int v9; // eax
  unsigned int v10; // r13d
  __int64 *v11; // rax
  unsigned __int64 v12; // r12
  __int64 v13; // rax
  __int64 v14; // rax
  __int64 v15; // r12
  __int64 v16; // r13
  __int64 v17; // r14
  _BYTE *v18; // rax
  __int64 v19; // rax
  _BYTE *v20; // r15
  __int64 (__fastcall *v21)(__int64, __int64, __int64); // rax
  unsigned int *v22; // r13
  unsigned int *v23; // rbx
  __int64 v24; // rdx
  unsigned int v25; // esi
  _QWORD *v26; // rax
  __int64 v27; // r9
  __int64 v28; // r12
  unsigned int *v29; // r13
  unsigned int *v30; // rbx
  __int64 v31; // rdx
  unsigned int v32; // esi
  __int64 v33; // r13
  __int64 v34; // rax
  _BYTE *v35; // r12
  _QWORD *v36; // rax
  __int64 v37; // r15
  unsigned __int64 v38; // r14
  _BYTE *v39; // rbx
  __int64 v40; // rdx
  unsigned int v41; // esi
  __int64 v42; // rsi
  __int64 (__fastcall *v43)(__int64, __int64, __int64); // rax
  unsigned __int64 v44; // r14
  _BYTE *v45; // rbx
  __int64 v46; // rdx
  unsigned int v47; // esi
  unsigned __int64 v48; // r14
  _BYTE *v49; // rbx
  __int64 v50; // rdx
  unsigned int v51; // esi
  _QWORD *v52; // rax
  __int64 v53; // rbx
  unsigned int *v54; // r13
  unsigned int *v55; // r12
  __int64 v56; // rdx
  unsigned int v57; // esi
  __int64 **v58; // rax
  unsigned __int64 v59; // rax
  unsigned int *v61; // r13
  unsigned int *v62; // rbx
  __int64 v63; // rdx
  unsigned int v64; // esi
  __int64 v65; // rbx
  unsigned __int64 v66; // rdx
  __int64 v67; // rax
  __int64 v68; // r12
  unsigned __int64 v69; // r8
  __int64 v70; // [rsp+28h] [rbp-348h]
  __int64 *v71; // [rsp+30h] [rbp-340h]
  __int64 **v72; // [rsp+38h] [rbp-338h]
  __int64 v73; // [rsp+40h] [rbp-330h]
  _QWORD *v74; // [rsp+48h] [rbp-328h]
  __int64 *v75; // [rsp+58h] [rbp-318h]
  __int64 v76; // [rsp+60h] [rbp-310h]
  __int64 *v77; // [rsp+68h] [rbp-308h]
  unsigned __int8 v78; // [rsp+9Ch] [rbp-2D4h]
  char v79; // [rsp+9Dh] [rbp-2D3h]
  __int64 *v80; // [rsp+A0h] [rbp-2D0h]
  _QWORD *v81; // [rsp+A8h] [rbp-2C8h]
  _BYTE v82[32]; // [rsp+B0h] [rbp-2C0h] BYREF
  __int16 v83; // [rsp+D0h] [rbp-2A0h]
  _BYTE v84[32]; // [rsp+E0h] [rbp-290h] BYREF
  __int16 v85; // [rsp+100h] [rbp-270h]
  unsigned int *v86; // [rsp+110h] [rbp-260h] BYREF
  __int64 v87; // [rsp+118h] [rbp-258h]
  _BYTE v88[32]; // [rsp+120h] [rbp-250h] BYREF
  __int64 v89; // [rsp+140h] [rbp-230h]
  __int64 v90; // [rsp+148h] [rbp-228h]
  __int64 v91; // [rsp+150h] [rbp-220h]
  __int64 v92; // [rsp+158h] [rbp-218h]
  void **v93; // [rsp+160h] [rbp-210h]
  void **v94; // [rsp+168h] [rbp-208h]
  __int64 v95; // [rsp+170h] [rbp-200h]
  int v96; // [rsp+178h] [rbp-1F8h]
  __int16 v97; // [rsp+17Ch] [rbp-1F4h]
  char v98; // [rsp+17Eh] [rbp-1F2h]
  __int64 v99; // [rsp+180h] [rbp-1F0h]
  __int64 v100; // [rsp+188h] [rbp-1E8h]
  void *v101; // [rsp+190h] [rbp-1E0h] BYREF
  void *v102; // [rsp+198h] [rbp-1D8h] BYREF
  _BYTE *v103; // [rsp+1A0h] [rbp-1D0h] BYREF
  __int64 v104; // [rsp+1A8h] [rbp-1C8h]
  _BYTE v105[16]; // [rsp+1B0h] [rbp-1C0h] BYREF
  __int16 v106; // [rsp+1C0h] [rbp-1B0h]
  __int64 v107; // [rsp+1D0h] [rbp-1A0h]
  __int64 v108; // [rsp+1D8h] [rbp-198h]
  __int64 v109; // [rsp+1E0h] [rbp-190h]
  __int64 v110; // [rsp+1E8h] [rbp-188h]
  void **v111; // [rsp+1F0h] [rbp-180h]
  void **v112; // [rsp+1F8h] [rbp-178h]
  __int64 v113; // [rsp+200h] [rbp-170h]
  int v114; // [rsp+208h] [rbp-168h]
  __int16 v115; // [rsp+20Ch] [rbp-164h]
  char v116; // [rsp+20Eh] [rbp-162h]
  __int64 v117; // [rsp+210h] [rbp-160h]
  __int64 v118; // [rsp+218h] [rbp-158h]
  void *v119; // [rsp+220h] [rbp-150h] BYREF
  void *v120; // [rsp+228h] [rbp-148h] BYREF
  __int64 *v121; // [rsp+230h] [rbp-140h] BYREF
  __int64 v122; // [rsp+238h] [rbp-138h]
  _BYTE v123[304]; // [rsp+240h] [rbp-130h] BYREF

  v6 = a3;
  v121 = (__int64 *)v123;
  v122 = 0x2000000000LL;
  if ( a3 )
  {
    v7 = a2;
    v75 = &a2[a3];
  }
  else
  {
    v65 = *(_QWORD *)(a1 + 64);
    if ( v65 == a1 + 56 )
    {
      v7 = (__int64 *)v123;
      v75 = (__int64 *)v123;
    }
    else
    {
      v66 = 32;
      v67 = 0;
      while ( 1 )
      {
        v68 = v65 - 56;
        v69 = v67 + 1;
        if ( !v65 )
          v68 = 0;
        if ( v69 > v66 )
        {
          sub_C8D5F0((__int64)&v121, v123, v67 + 1, 8u, v69, a6);
          v67 = (unsigned int)v122;
        }
        v121[v67] = v68;
        v67 = (unsigned int)(v122 + 1);
        LODWORD(v122) = v122 + 1;
        v65 = *(_QWORD *)(v65 + 8);
        if ( v65 == a1 + 56 )
          break;
        v66 = HIDWORD(v122);
      }
      v7 = v121;
      v6 = (unsigned int)v67;
      v75 = &v121[(unsigned int)v67];
    }
  }
  v71 = *(__int64 **)a1;
  v77 = (__int64 *)sub_BCE3C0(*(__int64 **)a1, *(_DWORD *)(a1 + 320));
  v72 = (__int64 **)sub_BCD420(v77, v6);
  v79 = sub_AE5020(a1 + 312, (__int64)v77);
  v106 = 257;
  v8 = sub_ACADE0(v72);
  v9 = *(_DWORD *)(a1 + 324);
  BYTE4(v86) = 1;
  LODWORD(v86) = v9;
  v74 = sub_BD2C40(88, unk_3F0FAE8);
  if ( v74 )
    sub_B30000((__int64)v74, a1, v72, 0, 7, v8, (__int64)&v103, 0, 0, (__int64)v86, 0);
  sub_B2F770((__int64)v74, v79);
  v106 = 257;
  v10 = *(_DWORD *)(a1 + 320);
  v11 = (__int64 *)sub_BCB120(v71);
  v12 = sub_BCF640(v11, 0);
  v13 = sub_BD2DA0(136);
  v70 = v13;
  if ( v13 )
    sub_B2C3B0(v13, v12, 7, v10, (__int64)&v103, a1);
  v106 = 257;
  v14 = sub_22077B0(0x50u);
  v15 = v14;
  if ( v14 )
    sub_AA4D50(v14, (__int64)v71, (__int64)&v103, v70, 0);
  v89 = v15;
  v16 = sub_AA48A0(v15);
  LOWORD(v91) = 0;
  v86 = (unsigned int *)v88;
  v87 = 0x200000000LL;
  v93 = &v101;
  v94 = &v102;
  v92 = v16;
  v97 = 512;
  v101 = &unk_49DA100;
  v95 = 0;
  v96 = 0;
  v98 = 7;
  v99 = 0;
  v100 = 0;
  v102 = &unk_49DA0B0;
  v90 = v15 + 48;
  if ( v7 == v75 )
  {
    v78 = 0;
  }
  else
  {
    v80 = v7;
    v17 = 0;
    v78 = 0;
    do
    {
      v81 = (_QWORD *)*v80;
      v18 = sub_B30850(*v80);
      if ( *(_DWORD *)(*((_QWORD *)v18 + 3) + 12LL) == 1 )
      {
        v106 = 257;
        v19 = sub_921880(&v86, *((_QWORD *)v18 + 3), (int)v18, 0, 0, (__int64)&v103, 0);
        v85 = 257;
        v20 = (_BYTE *)v19;
        if ( v77 != *(__int64 **)(v19 + 8) )
        {
          if ( *(_BYTE *)v19 > 0x15u )
          {
            v106 = 257;
            v20 = (_BYTE *)sub_B52210(v19, (__int64)v77, (__int64)&v103, 0, 0);
            (*((void (__fastcall **)(void **, _BYTE *, _BYTE *, __int64, __int64))*v94 + 2))(v94, v20, v84, v90, v91);
            v61 = v86;
            v62 = &v86[4 * (unsigned int)v87];
            if ( v86 != v62 )
            {
              do
              {
                v63 = *((_QWORD *)v61 + 1);
                v64 = *v61;
                v61 += 4;
                sub_B99FD0((__int64)v20, v64, v63);
              }
              while ( v62 != v61 );
            }
          }
          else
          {
            v21 = (__int64 (__fastcall *)(__int64, __int64, __int64))*((_QWORD *)*v93 + 17);
            if ( v21 == sub_928970 )
              v20 = (_BYTE *)sub_ADAFB0((unsigned __int64)v20, (__int64)v77);
            else
              v20 = (_BYTE *)v21((__int64)v93, (__int64)v20, (__int64)v77);
            if ( *v20 > 0x1Cu )
            {
              (*((void (__fastcall **)(void **, _BYTE *, _BYTE *, __int64, __int64))*v94 + 2))(v94, v20, v84, v90, v91);
              v22 = v86;
              v23 = &v86[4 * (unsigned int)v87];
              if ( v86 != v23 )
              {
                do
                {
                  v24 = *((_QWORD *)v22 + 1);
                  v25 = *v22;
                  v22 += 4;
                  sub_B99FD0((__int64)v20, v25, v24);
                }
                while ( v23 != v22 );
              }
            }
          }
        }
        v106 = 257;
        v73 = v17 + 1;
        v76 = sub_24DBB60((__int64 *)&v86, (__int64)v72, (__int64)v74, 0, v17, (__int64)&v103);
        v106 = 257;
        v26 = sub_BD2C40(80, unk_3F10A10);
        v28 = (__int64)v26;
        if ( v26 )
          sub_B4D3C0((__int64)v26, (__int64)v20, v76, 0, v79, v27, 0, 0);
        (*((void (__fastcall **)(void **, __int64, _BYTE **, __int64, __int64))*v94 + 2))(v94, v28, &v103, v90, v91);
        v29 = v86;
        v30 = &v86[4 * (unsigned int)v87];
        if ( v86 != v30 )
        {
          do
          {
            v31 = *((_QWORD *)v29 + 1);
            v32 = *v29;
            v29 += 4;
            sub_B99FD0(v28, v32, v31);
          }
          while ( v30 != v29 );
        }
        v33 = v81[2];
        if ( !v33 )
          goto LABEL_42;
        do
        {
          v34 = v33;
          v33 = *(_QWORD *)(v33 + 8);
          v35 = *(_BYTE **)(v34 + 24);
          if ( *v35 <= 0x1Cu )
          {
            v78 = 1;
          }
          else
          {
            v110 = sub_BD5C60(*(_QWORD *)(v34 + 24));
            v111 = &v119;
            v112 = &v120;
            v103 = v105;
            v119 = &unk_49DA100;
            v104 = 0x200000000LL;
            v115 = 512;
            LOWORD(v109) = 0;
            v120 = &unk_49DA0B0;
            v113 = 0;
            v114 = 0;
            v116 = 7;
            v117 = 0;
            v118 = 0;
            v107 = 0;
            v108 = 0;
            sub_D5F1F0((__int64)&v103, (__int64)v35);
            v83 = 257;
            v85 = 257;
            v36 = sub_BD2C40(80, 1u);
            v37 = (__int64)v36;
            if ( v36 )
              sub_B4D190((__int64)v36, (__int64)v77, v76, (__int64)v84, 0, v79, 0, 0);
            (*((void (__fastcall **)(void **, __int64, _BYTE *, __int64, __int64))*v112 + 2))(
              v112,
              v37,
              v82,
              v108,
              v109);
            v38 = (unsigned __int64)v103;
            v39 = &v103[16 * (unsigned int)v104];
            if ( v103 != v39 )
            {
              do
              {
                v40 = *(_QWORD *)(v38 + 8);
                v41 = *(_DWORD *)v38;
                v38 += 16LL;
                sub_B99FD0(v37, v41, v40);
              }
              while ( v39 != (_BYTE *)v38 );
            }
            v83 = 257;
            v42 = v81[1];
            if ( v42 != *(_QWORD *)(v37 + 8) )
            {
              if ( *(_BYTE *)v37 > 0x15u )
              {
                v85 = 257;
                v37 = sub_B52210(v37, v42, (__int64)v84, 0, 0);
                (*((void (__fastcall **)(void **, __int64, _BYTE *, __int64, __int64))*v112 + 2))(
                  v112,
                  v37,
                  v82,
                  v108,
                  v109);
                v48 = (unsigned __int64)v103;
                v49 = &v103[16 * (unsigned int)v104];
                if ( v103 != v49 )
                {
                  do
                  {
                    v50 = *(_QWORD *)(v48 + 8);
                    v51 = *(_DWORD *)v48;
                    v48 += 16LL;
                    sub_B99FD0(v37, v51, v50);
                  }
                  while ( v49 != (_BYTE *)v48 );
                }
              }
              else
              {
                v43 = (__int64 (__fastcall *)(__int64, __int64, __int64))*((_QWORD *)*v111 + 17);
                if ( v43 == sub_928970 )
                  v37 = sub_ADAFB0(v37, v42);
                else
                  v37 = v43((__int64)v111, v37, v81[1]);
                if ( *(_BYTE *)v37 > 0x1Cu )
                {
                  (*((void (__fastcall **)(void **, __int64, _BYTE *, __int64, __int64))*v112 + 2))(
                    v112,
                    v37,
                    v82,
                    v108,
                    v109);
                  v44 = (unsigned __int64)v103;
                  v45 = &v103[16 * (unsigned int)v104];
                  if ( v103 != v45 )
                  {
                    do
                    {
                      v46 = *(_QWORD *)(v44 + 8);
                      v47 = *(_DWORD *)v44;
                      v44 += 16LL;
                      sub_B99FD0(v37, v47, v46);
                    }
                    while ( v45 != (_BYTE *)v44 );
                  }
                }
              }
            }
            sub_BD2ED0((__int64)v35, (__int64)v81, v37);
            nullsub_61();
            v119 = &unk_49DA100;
            nullsub_63();
            if ( v103 != v105 )
              _libc_free((unsigned __int64)v103);
          }
        }
        while ( v33 );
        v17 = v73;
        if ( !v81[2] )
        {
LABEL_42:
          sub_B307B0(v81);
          v17 = v73;
        }
      }
      else
      {
        v78 = 1;
      }
      ++v80;
    }
    while ( v75 != v80 );
    v16 = v92;
  }
  v106 = 257;
  v52 = sub_BD2C40(72, 0);
  v53 = (__int64)v52;
  if ( v52 )
    sub_B4BB80((__int64)v52, v16, 0, 0, 0, 0);
  (*((void (__fastcall **)(void **, __int64, _BYTE **, __int64, __int64))*v94 + 2))(v94, v53, &v103, v90, v91);
  v54 = v86;
  v55 = &v86[4 * (unsigned int)v87];
  if ( v86 != v55 )
  {
    do
    {
      v56 = *((_QWORD *)v54 + 1);
      v57 = *v54;
      v54 += 4;
      sub_B99FD0(v53, v57, v56);
    }
    while ( v55 != v54 );
  }
  v58 = (__int64 **)sub_BCE3C0(v71, 0);
  v59 = sub_AC9EC0(v58);
  sub_2A3ED40((__int64 **)a1, v70, 0xAu, v59);
  nullsub_61();
  v101 = &unk_49DA100;
  nullsub_63();
  if ( v86 != (unsigned int *)v88 )
    _libc_free((unsigned __int64)v86);
  if ( v121 != (__int64 *)v123 )
    _libc_free((unsigned __int64)v121);
  return v78;
}
