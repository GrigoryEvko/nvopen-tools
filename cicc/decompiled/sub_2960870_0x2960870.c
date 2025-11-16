// Function: sub_2960870
// Address: 0x2960870
//
__int64 __fastcall sub_2960870(
        __int64 a1,
        __int64 *a2,
        __int64 a3,
        char a4,
        __int64 a5,
        __int64 a6,
        __int64 a7,
        _QWORD *a8)
{
  unsigned int v9; // eax
  _BYTE *v10; // r14
  unsigned __int8 *v11; // r12
  __int64 v12; // rax
  __int64 v13; // rdx
  __int64 v14; // rcx
  __int64 v15; // r8
  __int64 v16; // r9
  _QWORD *v17; // r15
  __int64 v18; // rax
  unsigned __int8 *v19; // rax
  int v20; // ecx
  __int64 v21; // rsi
  int v22; // ecx
  unsigned int v23; // edx
  _QWORD *v24; // rax
  _BYTE *v25; // rdi
  _BYTE *v26; // rax
  __int64 i; // r14
  __int64 v28; // rsi
  _QWORD *v29; // rax
  _QWORD *v30; // rdx
  char v31; // al
  __int64 v32; // rax
  __int64 v33; // rdi
  __int64 v34; // rcx
  __int64 v35; // rax
  __int64 v36; // rax
  __int64 v37; // rax
  __int64 v38; // rax
  _QWORD *v39; // rbx
  __int64 v40; // r13
  __int64 v41; // rbx
  _QWORD *v42; // rax
  __int64 v43; // r9
  __int64 v44; // r14
  __int64 v45; // r13
  __int64 *v46; // rbx
  __int64 v47; // rdx
  unsigned int v48; // esi
  unsigned int v49; // eax
  __int64 *v51; // rdx
  __int64 *v52; // r14
  int v53; // esi
  unsigned __int64 *v54; // r15
  int v55; // eax
  __int64 v56; // rdx
  unsigned __int64 *v57; // rdi
  __int64 *v58; // rdx
  int v59; // eax
  int v60; // r8d
  _QWORD *v61; // rbx
  _QWORD *v62; // r13
  __int64 v63; // rdx
  __int64 v64; // rax
  _QWORD *v65; // rdi
  int v66; // esi
  unsigned __int64 *v67; // rbx
  int v68; // eax
  bool v69; // zf
  unsigned __int64 *v70; // rdi
  __int64 *v71; // rax
  __int64 v72; // rax
  _QWORD *v73; // rbx
  _QWORD *v74; // r13
  __int64 v75; // rsi
  unsigned __int64 *v79; // [rsp+38h] [rbp-198h]
  __int64 v82; // [rsp+70h] [rbp-160h]
  __int64 *v83; // [rsp+78h] [rbp-158h]
  unsigned __int64 *v84; // [rsp+80h] [rbp-150h] BYREF
  unsigned __int64 *v85; // [rsp+88h] [rbp-148h] BYREF
  unsigned __int64 *v86; // [rsp+90h] [rbp-140h] BYREF
  __int64 v87; // [rsp+98h] [rbp-138h] BYREF
  __int64 v88; // [rsp+A0h] [rbp-130h]
  __int64 v89; // [rsp+A8h] [rbp-128h]
  __int64 *v90; // [rsp+B0h] [rbp-120h]
  __int64 v91; // [rsp+C0h] [rbp-110h] BYREF
  _QWORD *v92; // [rsp+C8h] [rbp-108h]
  __int64 v93; // [rsp+D0h] [rbp-100h]
  unsigned int v94; // [rsp+D8h] [rbp-F8h]
  _QWORD *v95; // [rsp+E8h] [rbp-E8h]
  unsigned int v96; // [rsp+F8h] [rbp-D8h]
  char v97; // [rsp+100h] [rbp-D0h]
  __int64 *v98; // [rsp+110h] [rbp-C0h] BYREF
  __int64 v99; // [rsp+118h] [rbp-B8h] BYREF
  __int64 v100; // [rsp+120h] [rbp-B0h] BYREF
  __int64 v101; // [rsp+128h] [rbp-A8h]
  __int64 *v102; // [rsp+130h] [rbp-A0h]
  __int64 v103; // [rsp+140h] [rbp-90h]
  unsigned __int64 *v104; // [rsp+148h] [rbp-88h]
  __int64 v105; // [rsp+150h] [rbp-80h]
  __int64 v106; // [rsp+158h] [rbp-78h]
  void **v107; // [rsp+160h] [rbp-70h]
  void **v108; // [rsp+168h] [rbp-68h]
  __int64 v109; // [rsp+170h] [rbp-60h]
  int v110; // [rsp+178h] [rbp-58h]
  __int16 v111; // [rsp+17Ch] [rbp-54h]
  char v112; // [rsp+17Eh] [rbp-52h]
  __int64 v113; // [rsp+180h] [rbp-50h]
  __int64 v114; // [rsp+188h] [rbp-48h]
  void *v115; // [rsp+190h] [rbp-40h] BYREF
  void *v116; // [rsp+198h] [rbp-38h] BYREF

  v91 = 0;
  v9 = sub_AF1560(0x56u);
  v94 = v9;
  if ( v9 )
  {
    v92 = (_QWORD *)sub_C7D670((unsigned __int64)v9 << 6, 8);
    sub_23FE7B0((__int64)&v91);
  }
  else
  {
    v92 = 0;
    v93 = 0;
  }
  v97 = 0;
  v83 = &a2[a3];
  v79 = (unsigned __int64 *)(a1 + 48);
  if ( a2 != v83 )
  {
    while ( 1 )
    {
      v10 = (_BYTE *)*(v83 - 1);
      v11 = (unsigned __int8 *)sub_B47F80(v10);
      v12 = v82;
      LOWORD(v12) = 0;
      v82 = v12;
      sub_B44240(v11, a1, v79, v12);
      sub_FC75A0((__int64 *)&v98, (__int64)&v91, 3, 0, 0, 0);
      sub_FCD280((__int64 *)&v98, v11, v13, v14, v15, v16);
      sub_FC7680((__int64 *)&v98, (__int64)v11);
      v99 = 2;
      v100 = 0;
      v101 = (__int64)v10;
      if ( v10 != 0 && v10 + 4096 != 0 && v10 != (_BYTE *)-8192LL )
        sub_BD73F0((__int64)&v99);
      v98 = (__int64 *)&unk_49DD7B0;
      v102 = &v91;
      if ( !(unsigned __int8)sub_F9E960((__int64)&v91, (__int64)&v98, &v85) )
        break;
      v17 = v85 + 5;
      v18 = v101;
LABEL_9:
      v98 = (__int64 *)&unk_49DB368;
      if ( v18 != 0 && v18 != -4096 && v18 != -8192 )
        sub_BD60C0(&v99);
      v19 = (unsigned __int8 *)v17[2];
      if ( v11 != v19 )
      {
        if ( v19 != 0 && v19 + 4096 != 0 && v19 != (unsigned __int8 *)-8192LL )
          sub_BD60C0(v17);
        v17[2] = v11;
        if ( v11 + 4096 != 0 && v11 != 0 && v11 != (unsigned __int8 *)-8192LL )
          sub_BD73F0((__int64)v17);
      }
      if ( a8 )
      {
        v20 = *(_DWORD *)(*a8 + 56LL);
        v21 = *(_QWORD *)(*a8 + 40LL);
        if ( v20 )
        {
          v22 = v20 - 1;
          v23 = v22 & (((unsigned int)v10 >> 9) ^ ((unsigned int)v10 >> 4));
          v24 = (_QWORD *)(v21 + 16LL * v23);
          v25 = (_BYTE *)*v24;
          if ( v10 == (_BYTE *)*v24 )
          {
LABEL_22:
            v26 = (_BYTE *)v24[1];
            if ( v26 && *v26 == 26 )
            {
              for ( i = *((_QWORD *)v26 - 4); ; i = *v52 )
              {
                v28 = *(_QWORD *)(i + 64);
                if ( *(_BYTE *)(a7 + 84) )
                  break;
                while ( 1 )
                {
                  if ( !sub_C8CA60(a7 + 56, v28) )
                    goto LABEL_38;
LABEL_30:
                  v31 = *(_BYTE *)i;
                  if ( *(_BYTE *)i != 28 )
                    break;
                  v32 = sub_D4B130(a7);
                  v33 = *(_QWORD *)(i - 8);
                  v34 = v32;
                  if ( (*(_DWORD *)(i + 4) & 0x7FFFFFF) != 0 )
                  {
                    v35 = 0;
                    while ( v34 != *(_QWORD *)(v33 + 32LL * *(unsigned int *)(i + 76) + 8 * v35) )
                    {
                      if ( (*(_DWORD *)(i + 4) & 0x7FFFFFF) == (_DWORD)++v35 )
                        goto LABEL_59;
                    }
                    v36 = 32 * v35;
                  }
                  else
                  {
LABEL_59:
                    v36 = 0x1FFFFFFFE0LL;
                  }
                  i = *(_QWORD *)(v33 + v36);
                  v28 = *(_QWORD *)(i + 64);
                  if ( *(_BYTE *)(a7 + 84) )
                    goto LABEL_26;
                }
                v51 = (__int64 *)(i - 32);
                v52 = (__int64 *)(i - 64);
                if ( v31 == 26 )
                  v52 = v51;
              }
LABEL_26:
              v29 = *(_QWORD **)(a7 + 64);
              v30 = &v29[*(unsigned int *)(a7 + 76)];
              if ( v29 != v30 )
              {
                while ( v28 != *v29 )
                {
                  if ( v30 == ++v29 )
                    goto LABEL_38;
                }
                goto LABEL_30;
              }
LABEL_38:
              sub_D694D0(a8, (__int64)v11, i, *((_QWORD *)v11 + 5), 2u, 1u);
            }
          }
          else
          {
            v59 = 1;
            while ( v25 != (_BYTE *)-4096LL )
            {
              v60 = v59 + 1;
              v23 = v22 & (v59 + v23);
              v24 = (_QWORD *)(v21 + 16LL * v23);
              v25 = (_BYTE *)*v24;
              if ( v10 == (_BYTE *)*v24 )
                goto LABEL_22;
              v59 = v60;
            }
          }
        }
      }
      if ( a2 == --v83 )
        goto LABEL_40;
    }
    v53 = v94;
    v54 = v85;
    ++v91;
    v55 = v93 + 1;
    v86 = v85;
    if ( 4 * ((int)v93 + 1) >= 3 * v94 )
    {
      v53 = 2 * v94;
    }
    else if ( v94 - HIDWORD(v93) - v55 > v94 >> 3 )
    {
      goto LABEL_62;
    }
    sub_CF32C0((__int64)&v91, v53);
    sub_F9E960((__int64)&v91, (__int64)&v98, &v86);
    v54 = v86;
    v55 = v93 + 1;
LABEL_62:
    LODWORD(v93) = v55;
    v18 = v54[3];
    if ( v18 == -4096 )
    {
      v56 = v101;
      v57 = v54 + 1;
      if ( v101 != -4096 )
      {
LABEL_67:
        v54[3] = v56;
        if ( v56 != -4096 && v56 != 0 && v56 != -8192 )
          sub_BD6050(v57, v99 & 0xFFFFFFFFFFFFFFF8LL);
        v18 = v101;
      }
    }
    else
    {
      v56 = v101;
      --HIDWORD(v93);
      if ( v101 != v18 )
      {
        v57 = v54 + 1;
        if ( v18 && v18 != -8192 )
        {
          sub_BD60C0(v57);
          v56 = v101;
          v57 = v54 + 1;
        }
        goto LABEL_67;
      }
    }
    v58 = v102;
    v17 = v54 + 5;
    *v17 = 6;
    v17[1] = 0;
    *(v17 - 1) = v58;
    v17[2] = 0;
    goto LABEL_9;
  }
LABEL_40:
  v37 = sub_AA48A0(a1);
  v108 = &v116;
  v106 = v37;
  v107 = &v115;
  v98 = &v100;
  v115 = &unk_49DA100;
  v99 = 0x200000000LL;
  LOWORD(v105) = 0;
  v116 = &unk_49DA0B0;
  v109 = 0;
  v104 = v79;
  v110 = 0;
  v38 = *a2;
  v111 = 512;
  v112 = 7;
  v113 = 0;
  v114 = 0;
  v103 = a1;
  v87 = 2;
  v88 = 0;
  v89 = v38;
  if ( v38 != -4096 && v38 != 0 && v38 != -8192 )
    sub_BD73F0((__int64)&v87);
  v90 = &v91;
  v86 = (unsigned __int64 *)&unk_49DD7B0;
  if ( (unsigned __int8)sub_F9E960((__int64)&v91, (__int64)&v86, &v84) )
  {
    v39 = v84 + 5;
    goto LABEL_45;
  }
  v66 = v94;
  v67 = v84;
  ++v91;
  v68 = v93 + 1;
  v85 = v84;
  if ( 4 * ((int)v93 + 1) >= 3 * v94 )
  {
    v66 = 2 * v94;
    goto LABEL_104;
  }
  if ( v94 - HIDWORD(v93) - v68 <= v94 >> 3 )
  {
LABEL_104:
    sub_CF32C0((__int64)&v91, v66);
    sub_F9E960((__int64)&v91, (__int64)&v86, &v85);
    v67 = v85;
    v68 = v93 + 1;
  }
  v69 = v67[3] == -4096;
  LODWORD(v93) = v68;
  if ( !v69 )
    --HIDWORD(v93);
  v70 = v67 + 1;
  v39 = v67 + 5;
  sub_2957970(v70, &v87);
  v71 = v90;
  *v39 = 6;
  v39[1] = 0;
  *(v39 - 1) = v71;
  v39[2] = 0;
LABEL_45:
  v86 = (unsigned __int64 *)&unk_49DB368;
  sub_D68D70(&v87);
  v40 = v39[2];
  if ( a4 )
  {
    v41 = a5;
    a5 = a6;
    a6 = v41;
  }
  LOWORD(v90) = 257;
  v42 = sub_BD2C40(72, 3u);
  v44 = (__int64)v42;
  if ( v42 )
    sub_B4C9A0((__int64)v42, a6, a5, v40, 3u, v43, 0, 0);
  (*((void (__fastcall **)(void **, __int64, unsigned __int64 **, unsigned __int64 *, __int64))*v108 + 2))(
    v108,
    v44,
    &v86,
    v104,
    v105);
  v45 = (__int64)v98;
  v46 = &v98[2 * (unsigned int)v99];
  if ( v98 != v46 )
  {
    do
    {
      v47 = *(_QWORD *)(v45 + 8);
      v48 = *(_DWORD *)v45;
      v45 += 16;
      sub_B99FD0(v44, v48, v47);
    }
    while ( v46 != (__int64 *)v45 );
  }
  nullsub_61();
  v115 = &unk_49DA100;
  nullsub_63();
  if ( v98 != &v100 )
    _libc_free((unsigned __int64)v98);
  if ( v97 )
  {
    v72 = v96;
    v97 = 0;
    if ( v96 )
    {
      v73 = v95;
      v74 = &v95[2 * v96];
      do
      {
        if ( *v73 != -4096 && *v73 != -8192 )
        {
          v75 = v73[1];
          if ( v75 )
            sub_B91220((__int64)(v73 + 1), v75);
        }
        v73 += 2;
      }
      while ( v74 != v73 );
      v72 = v96;
    }
    sub_C7D6A0((__int64)v95, 16 * v72, 8);
  }
  v49 = v94;
  if ( v94 )
  {
    v61 = v92;
    v87 = 2;
    v88 = 0;
    v62 = &v92[8 * (unsigned __int64)v94];
    v89 = -4096;
    v86 = (unsigned __int64 *)&unk_49DD7B0;
    v98 = (__int64 *)&unk_49DD7B0;
    v63 = -4096;
    v90 = 0;
    v99 = 2;
    v100 = 0;
    v101 = -8192;
    v102 = 0;
    while ( 1 )
    {
      v64 = v61[3];
      if ( v64 != v63 && v64 != v101 )
        sub_D68D70(v61 + 5);
      *v61 = &unk_49DB368;
      v65 = v61 + 1;
      v61 += 8;
      sub_D68D70(v65);
      if ( v62 == v61 )
        break;
      v63 = v89;
    }
    v98 = (__int64 *)&unk_49DB368;
    sub_D68D70(&v99);
    v86 = (unsigned __int64 *)&unk_49DB368;
    sub_D68D70(&v87);
    v49 = v94;
  }
  return sub_C7D6A0((__int64)v92, (unsigned __int64)v49 << 6, 8);
}
