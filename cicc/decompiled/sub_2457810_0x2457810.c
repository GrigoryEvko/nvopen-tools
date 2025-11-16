// Function: sub_2457810
// Address: 0x2457810
//
__int64 __fastcall sub_2457810(__int64 a1)
{
  __int64 result; // rax
  __int64 v3; // rbx
  __int64 v4; // r13
  __int64 **v5; // r12
  _BYTE *v6; // r14
  __int64 v7; // rax
  unsigned __int8 v8; // r12
  __int64 v9; // rax
  __int64 v10; // rdx
  unsigned __int64 v11; // rax
  _QWORD *v12; // rax
  __int64 v13; // rbx
  unsigned int *v14; // r13
  unsigned int *v15; // r12
  __int64 v16; // rdx
  unsigned int v17; // esi
  __int64 v18; // rax
  char v19; // al
  __int16 v20; // cx
  _QWORD *v21; // rax
  _BYTE *v22; // rbx
  __int64 v23; // r12
  unsigned int *v24; // r12
  unsigned int *v25; // r13
  __int64 v26; // rdx
  unsigned int v27; // esi
  __int64 v28; // r13
  __int64 v29; // rax
  char v30; // al
  __int16 v31; // cx
  _QWORD *v32; // rax
  __int64 v33; // r9
  __int64 v34; // r12
  unsigned int *v35; // r14
  unsigned int *v36; // r13
  __int64 v37; // rdx
  unsigned int v38; // esi
  __int64 v39; // rax
  int v40; // ecx
  __int64 v41; // rsi
  int v42; // ecx
  __int64 v43; // rdx
  __int64 *v44; // rax
  __int64 v45; // rdi
  __int64 v46; // r13
  __int64 v47; // r14
  unsigned int v48; // esi
  __int64 v49; // rdi
  __int64 v50; // r9
  _QWORD *v51; // r10
  unsigned int v52; // edx
  __int64 *v53; // rax
  __int64 v54; // rcx
  unsigned __int64 v55; // rdx
  unsigned __int64 v56; // rsi
  __int64 v57; // rdi
  int v58; // ecx
  _QWORD *v59; // rdx
  _BYTE *v60; // rdi
  __int64 v61; // rbx
  __int64 v62; // r14
  unsigned int *v63; // r14
  unsigned int *v64; // r12
  __int64 v65; // rdx
  unsigned int v66; // esi
  __int64 v67; // r10
  __int64 (__fastcall *v68)(__int64, unsigned int, _BYTE *, __int64); // rax
  __int64 v69; // rax
  int v70; // ebx
  __int64 v71; // rbx
  unsigned int *v72; // rbx
  unsigned int *v73; // r12
  __int64 v74; // rdx
  unsigned int v75; // esi
  __int64 v76; // rax
  int v77; // eax
  int v78; // r8d
  _QWORD *v79; // rdx
  int v80; // eax
  int v81; // eax
  int v82; // r11d
  int v83; // r11d
  __int64 v84; // r9
  unsigned int v85; // edx
  __int64 v86; // rdi
  int v87; // esi
  _QWORD *v88; // rcx
  int v89; // r9d
  int v90; // r9d
  __int64 v91; // r8
  int v92; // ecx
  unsigned int v93; // r11d
  _QWORD *v94; // rdx
  __int64 v95; // rsi
  __int64 v96; // [rsp-180h] [rbp-180h]
  __int64 **v97; // [rsp-180h] [rbp-180h]
  __int64 **v98; // [rsp-180h] [rbp-180h]
  __int64 **v99; // [rsp-180h] [rbp-180h]
  __int64 v100; // [rsp-180h] [rbp-180h]
  __int64 v101; // [rsp-178h] [rbp-178h]
  __int64 v102; // [rsp-170h] [rbp-170h]
  unsigned int v103; // [rsp-170h] [rbp-170h]
  __int64 *v104; // [rsp-170h] [rbp-170h]
  __int16 v105; // [rsp-164h] [rbp-164h]
  __int16 v106; // [rsp-162h] [rbp-162h]
  __int64 v107; // [rsp-130h] [rbp-130h]
  _QWORD v108[4]; // [rsp-128h] [rbp-128h] BYREF
  __int16 v109; // [rsp-108h] [rbp-108h]
  _QWORD v110[4]; // [rsp-F8h] [rbp-F8h] BYREF
  __int16 v111; // [rsp-D8h] [rbp-D8h]
  unsigned int *v112; // [rsp-C8h] [rbp-C8h] BYREF
  __int64 v113; // [rsp-C0h] [rbp-C0h]
  _BYTE v114[32]; // [rsp-B8h] [rbp-B8h] BYREF
  __int64 v115; // [rsp-98h] [rbp-98h]
  __int64 v116; // [rsp-90h] [rbp-90h]
  __int64 v117; // [rsp-88h] [rbp-88h]
  __int64 v118; // [rsp-80h] [rbp-80h]
  void **v119; // [rsp-78h] [rbp-78h]
  void **v120; // [rsp-70h] [rbp-70h]
  __int64 v121; // [rsp-68h] [rbp-68h]
  int v122; // [rsp-60h] [rbp-60h]
  __int16 v123; // [rsp-5Ch] [rbp-5Ch]
  char v124; // [rsp-5Ah] [rbp-5Ah]
  __int64 v125; // [rsp-58h] [rbp-58h]
  __int64 v126; // [rsp-50h] [rbp-50h]
  void *v127; // [rsp-48h] [rbp-48h] BYREF
  void *v128; // [rsp-40h] [rbp-40h] BYREF

  result = *(_QWORD *)(a1 + 32);
  if ( (_DWORD)result )
  {
    v101 = 8LL * (unsigned int)result;
    v107 = 0;
    while ( 1 )
    {
      v102 = *(_QWORD *)(*(_QWORD *)(a1 + 24) + v107);
      v3 = *(_QWORD *)(*(_QWORD *)(a1 + 40) + v107);
      v4 = sub_11D7E40(*(__int64 **)(a1 + 8), v102);
      v5 = *(__int64 ***)(v4 + 8);
      v6 = *(_BYTE **)(*(_QWORD *)(a1 + 16) - 32LL);
      v118 = sub_BD5C60(v3);
      v112 = (unsigned int *)v114;
      v119 = &v127;
      v113 = 0x200000000LL;
      v120 = &v128;
      v123 = 512;
      LOWORD(v117) = 0;
      v121 = 0;
      v127 = &unk_49DA100;
      v122 = 0;
      v124 = 7;
      v125 = 0;
      v126 = 0;
      v115 = 0;
      v116 = 0;
      v128 = &unk_49DA0B0;
      sub_D5F1F0((__int64)&v112, v3);
      if ( !v6 || *v6 != 77 )
        goto LABEL_5;
      v60 = (_BYTE *)*((_QWORD *)v6 - 4);
      if ( (unsigned __int8)(*v60 - 42) >= 0x12u )
        v60 = 0;
      v111 = 257;
      v61 = sub_B47F80(v60);
      (*((void (__fastcall **)(void **, __int64, _QWORD *, __int64, __int64))*v120 + 2))(v120, v61, v110, v116, v117);
      v62 = 4LL * (unsigned int)v113;
      if ( v112 != &v112[v62] )
      {
        v97 = v5;
        v63 = &v112[v62];
        v64 = v112;
        do
        {
          v65 = *((_QWORD *)v64 + 1);
          v66 = *v64;
          v64 += 4;
          sub_B99FD0(v61, v66, v65);
        }
        while ( v63 != v64 );
        v5 = v97;
      }
      v109 = 257;
      v67 = sub_BCE3C0(*v5, 0);
      if ( v67 == *(_QWORD *)(v61 + 8) )
      {
        v6 = (_BYTE *)v61;
        goto LABEL_5;
      }
      v68 = (__int64 (__fastcall *)(__int64, unsigned int, _BYTE *, __int64))*((_QWORD *)*v119 + 15);
      if ( v68 != sub_920130 )
        break;
      if ( *(_BYTE *)v61 <= 0x15u )
      {
        v98 = (__int64 **)v67;
        if ( (unsigned __int8)sub_AC4810(0x30u) )
          v69 = sub_ADAB70(48, v61, v98, 0);
        else
          v69 = sub_AA93C0(0x30u, v61, (__int64)v98);
        v67 = (__int64)v98;
        v6 = (_BYTE *)v69;
LABEL_48:
        if ( v6 )
          goto LABEL_5;
      }
      v111 = 257;
      v6 = (_BYTE *)sub_B51D30(48, v61, v67, (__int64)v110, 0, 0);
      if ( (unsigned __int8)sub_920620((__int64)v6) )
      {
        v70 = v122;
        if ( v121 )
          sub_B99FD0((__int64)v6, 3u, v121);
        sub_B45150((__int64)v6, v70);
      }
      (*((void (__fastcall **)(void **, _BYTE *, _QWORD *, __int64, __int64))*v120 + 2))(v120, v6, v108, v116, v117);
      v71 = 4LL * (unsigned int)v113;
      if ( v112 != &v112[v71] )
      {
        v99 = v5;
        v72 = &v112[v71];
        v73 = v112;
        do
        {
          v74 = *((_QWORD *)v73 + 1);
          v75 = *v73;
          v73 += 4;
          sub_B99FD0((__int64)v6, v75, v74);
        }
        while ( v72 != v73 );
        v5 = v99;
      }
LABEL_5:
      if ( (_BYTE)qword_4FE6F28 )
      {
        v7 = sub_AA4E30(v115);
        v8 = -1;
        v9 = sub_9208B0(v7, *(_QWORD *)(v4 + 8));
        v110[1] = v10;
        v110[0] = (unsigned __int64)(v9 + 7) >> 3;
        v11 = sub_CA1930(v110);
        if ( v11 )
        {
          _BitScanReverse64(&v11, v11);
          v8 = 63 - (v11 ^ 0x3F);
        }
        v111 = 257;
        v12 = sub_BD2C40(80, unk_3F148C0);
        v13 = (__int64)v12;
        if ( v12 )
          sub_B4D750((__int64)v12, 1, (__int64)v6, v4, v8, 7, 1, 0, 0);
        (*((void (__fastcall **)(void **, __int64, _QWORD *, __int64, __int64))*v120 + 2))(v120, v13, v110, v116, v117);
        v14 = v112;
        v15 = &v112[4 * (unsigned int)v113];
        if ( v112 != v15 )
        {
          do
          {
            v16 = *((_QWORD *)v14 + 1);
            v17 = *v14;
            v14 += 4;
            sub_B99FD0(v13, v17, v16);
          }
          while ( v15 != v14 );
        }
      }
      else
      {
        v108[0] = "pgocount.promoted";
        v109 = 259;
        v18 = sub_AA4E30(v115);
        v19 = sub_AE5020(v18, (__int64)v5);
        HIBYTE(v20) = HIBYTE(v106);
        v111 = 257;
        LOBYTE(v20) = v19;
        v106 = v20;
        v21 = sub_BD2C40(80, unk_3F10A14);
        v22 = v21;
        if ( v21 )
          sub_B4D190((__int64)v21, (__int64)v5, (__int64)v6, (__int64)v110, 0, v106, 0, 0);
        (*((void (__fastcall **)(void **, _BYTE *, _QWORD *, __int64, __int64))*v120 + 2))(v120, v22, v108, v116, v117);
        v23 = 4LL * (unsigned int)v113;
        if ( v112 != &v112[v23] )
        {
          v96 = v4;
          v24 = &v112[v23];
          v25 = v112;
          do
          {
            v26 = *((_QWORD *)v25 + 1);
            v27 = *v25;
            v25 += 4;
            sub_B99FD0((__int64)v22, v27, v26);
          }
          while ( v24 != v25 );
          v4 = v96;
        }
        v111 = 257;
        v28 = sub_929C50(&v112, v22, (_BYTE *)v4, (__int64)v110, 0, 0);
        v29 = sub_AA4E30(v115);
        v30 = sub_AE5020(v29, *(_QWORD *)(v28 + 8));
        HIBYTE(v31) = HIBYTE(v105);
        v111 = 257;
        LOBYTE(v31) = v30;
        v105 = v31;
        v32 = sub_BD2C40(80, unk_3F10A10);
        v34 = (__int64)v32;
        if ( v32 )
          sub_B4D3C0((__int64)v32, v28, (__int64)v6, 0, v105, v33, 0, 0);
        (*((void (__fastcall **)(void **, __int64, _QWORD *, __int64, __int64))*v120 + 2))(v120, v34, v110, v116, v117);
        v35 = v112;
        v36 = &v112[4 * (unsigned int)v113];
        if ( v112 != v36 )
        {
          do
          {
            v37 = *((_QWORD *)v35 + 1);
            v38 = *v35;
            v35 += 4;
            sub_B99FD0(v34, v38, v37);
          }
          while ( v36 != v35 );
        }
        if ( (_BYTE)qword_4FE6828 )
        {
          v39 = *(_QWORD *)(a1 + 64);
          v40 = *(_DWORD *)(v39 + 24);
          v41 = *(_QWORD *)(v39 + 8);
          if ( v40 )
          {
            v42 = v40 - 1;
            v43 = v42 & (((unsigned int)v102 >> 9) ^ ((unsigned int)v102 >> 4));
            v44 = (__int64 *)(v41 + 16 * v43);
            v45 = *v44;
            if ( v102 != *v44 )
            {
              v77 = 1;
              while ( v45 != -4096 )
              {
                v78 = v77 + 1;
                LODWORD(v43) = v42 & (v77 + v43);
                v44 = (__int64 *)(v41 + 16LL * (unsigned int)v43);
                v45 = *v44;
                if ( v102 == *v44 )
                  goto LABEL_29;
                v77 = v78;
              }
              goto LABEL_12;
            }
LABEL_29:
            v46 = v44[1];
            if ( v46 )
            {
              v47 = *(_QWORD *)(a1 + 56);
              v48 = *(_DWORD *)(v47 + 24);
              if ( v48 )
              {
                v49 = *(_QWORD *)(v47 + 8);
                v50 = 1;
                v51 = 0;
                v52 = (v48 - 1) & (((unsigned int)v46 >> 4) ^ ((unsigned int)v46 >> 9));
                v53 = (__int64 *)(v49 + 152LL * v52);
                v54 = *v53;
                if ( v46 == *v53 )
                {
LABEL_32:
                  v55 = *((unsigned int *)v53 + 4);
                  v56 = *((unsigned int *)v53 + 5);
                  v57 = (__int64)(v53 + 1);
                  v58 = *((_DWORD *)v53 + 4);
                  if ( v55 >= v56 )
                  {
                    if ( v55 + 1 > v56 )
                    {
                      v104 = v53;
                      sub_C8D5F0(v57, v53 + 3, v55 + 1, 0x10u, v55 + 1, v50);
                      v53 = v104;
                      v55 = *((unsigned int *)v104 + 4);
                    }
                    v79 = (_QWORD *)(v53[1] + 16 * v55);
                    *v79 = v22;
                    v79[1] = v34;
                    ++*((_DWORD *)v53 + 4);
                    goto LABEL_12;
                  }
                  goto LABEL_33;
                }
                while ( v54 != -4096 )
                {
                  if ( v54 == -8192 && !v51 )
                    v51 = v53;
                  v52 = (v48 - 1) & (v50 + v52);
                  v53 = (__int64 *)(v49 + 152LL * v52);
                  v54 = *v53;
                  if ( v46 == *v53 )
                    goto LABEL_32;
                  v50 = (unsigned int)(v50 + 1);
                }
                if ( !v51 )
                  v51 = v53;
                v80 = *(_DWORD *)(v47 + 16);
                ++*(_QWORD *)v47;
                v81 = v80 + 1;
                if ( 4 * v81 < 3 * v48 )
                {
                  if ( v48 - *(_DWORD *)(v47 + 20) - v81 <= v48 >> 3 )
                  {
                    v103 = ((unsigned int)v46 >> 4) ^ ((unsigned int)v46 >> 9);
                    sub_2455850(v47, v48);
                    v89 = *(_DWORD *)(v47 + 24);
                    if ( !v89 )
                    {
LABEL_105:
                      ++*(_DWORD *)(v47 + 16);
                      BUG();
                    }
                    v90 = v89 - 1;
                    v91 = *(_QWORD *)(v47 + 8);
                    v92 = 1;
                    v93 = v90 & v103;
                    v94 = 0;
                    v51 = (_QWORD *)(v91 + 152LL * (v90 & v103));
                    v95 = *v51;
                    v81 = *(_DWORD *)(v47 + 16) + 1;
                    if ( v46 != *v51 )
                    {
                      while ( v95 != -4096 )
                      {
                        if ( !v94 && v95 == -8192 )
                          v94 = v51;
                        v93 = v90 & (v92 + v93);
                        v51 = (_QWORD *)(v91 + 152LL * v93);
                        v95 = *v51;
                        if ( v46 == *v51 )
                          goto LABEL_78;
                        ++v92;
                      }
                      if ( v94 )
                        v51 = v94;
                    }
                  }
                  goto LABEL_78;
                }
              }
              else
              {
                ++*(_QWORD *)v47;
              }
              sub_2455850(v47, 2 * v48);
              v82 = *(_DWORD *)(v47 + 24);
              if ( !v82 )
                goto LABEL_105;
              v83 = v82 - 1;
              v84 = *(_QWORD *)(v47 + 8);
              v85 = v83 & (((unsigned int)v46 >> 9) ^ ((unsigned int)v46 >> 4));
              v51 = (_QWORD *)(v84 + 152LL * v85);
              v86 = *v51;
              v81 = *(_DWORD *)(v47 + 16) + 1;
              if ( v46 != *v51 )
              {
                v87 = 1;
                v88 = 0;
                while ( v86 != -4096 )
                {
                  if ( !v88 && v86 == -8192 )
                    v88 = v51;
                  v85 = v83 & (v87 + v85);
                  v51 = (_QWORD *)(v84 + 152LL * v85);
                  v86 = *v51;
                  if ( v46 == *v51 )
                    goto LABEL_78;
                  ++v87;
                }
                if ( v88 )
                  v51 = v88;
              }
LABEL_78:
              *(_DWORD *)(v47 + 16) = v81;
              if ( *v51 != -4096 )
                --*(_DWORD *)(v47 + 20);
              *v51 = v46;
              v57 = (__int64)(v51 + 1);
              v58 = 0;
              v51[1] = v51 + 3;
              v55 = 0;
              v51[2] = 0x800000000LL;
LABEL_33:
              v59 = (_QWORD *)(*(_QWORD *)v57 + 16 * v55);
              if ( v59 )
              {
                *v59 = v22;
                v59[1] = v34;
                v58 = *(_DWORD *)(v57 + 8);
              }
              *(_DWORD *)(v57 + 8) = v58 + 1;
            }
          }
        }
      }
LABEL_12:
      nullsub_61();
      v127 = &unk_49DA100;
      nullsub_63();
      if ( v112 != (unsigned int *)v114 )
        _libc_free((unsigned __int64)v112);
      v107 += 8;
      result = v107;
      if ( v101 == v107 )
        return result;
    }
    v100 = v67;
    v76 = v68((__int64)v119, 48u, (_BYTE *)v61, v67);
    v67 = v100;
    v6 = (_BYTE *)v76;
    goto LABEL_48;
  }
  return result;
}
