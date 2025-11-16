// Function: sub_2CFA070
// Address: 0x2cfa070
//
unsigned __int64 __fastcall sub_2CFA070(__int64 a1, _QWORD *a2, __int64 a3, __int64 a4, __int16 a5, char a6)
{
  _QWORD *v6; // r15
  __int64 v7; // r14
  const char *v11; // rsi
  __int64 v12; // r9
  const char *v13; // r8
  unsigned int *v14; // rax
  int v15; // ecx
  unsigned int *v16; // rdx
  __int64 v17; // r10
  __int64 v18; // rax
  __int64 v19; // rsi
  unsigned int v20; // ecx
  __int64 v21; // rdx
  __int64 v22; // r8
  _QWORD *v23; // r11
  unsigned int v24; // r13d
  __int64 **v25; // rax
  __int64 **v26; // rax
  unsigned __int64 v27; // r13
  __int64 **v28; // rax
  __int64 **v29; // rax
  unsigned __int64 v30; // r13
  __int64 **v32; // rax
  int v33; // edx
  __int64 v34; // rdx
  unsigned __int8 v35; // bl
  _QWORD *v36; // rax
  __int16 v37; // bx
  _QWORD *v38; // rdx
  unsigned int v39; // esi
  int v40; // edx
  unsigned __int64 v41; // rax
  unsigned __int64 *v42; // rbx
  int v43; // ecx
  unsigned __int64 v44; // rdx
  unsigned __int64 *v45; // r13
  __int64 v46; // rdx
  _QWORD *v47; // rbx
  __int64 v48; // r9
  unsigned int v49; // r8d
  unsigned __int64 *v50; // rdx
  unsigned __int64 v51; // rcx
  int v52; // r9d
  int v53; // r8d
  __int64 v54; // r9
  int v55; // esi
  unsigned int v56; // edx
  unsigned __int64 *v57; // rcx
  unsigned __int64 v58; // rdi
  unsigned __int64 v59; // rsi
  int v60; // edi
  int v61; // edx
  int v62; // r8d
  __int64 v63; // r9
  int v64; // esi
  unsigned int v65; // edx
  unsigned __int64 v66; // rdi
  char v67; // [rsp+Ch] [rbp-154h]
  __int64 v68; // [rsp+10h] [rbp-150h]
  __int64 v70; // [rsp+38h] [rbp-128h]
  __int64 v71; // [rsp+38h] [rbp-128h]
  __int64 v72; // [rsp+38h] [rbp-128h]
  _QWORD *v73; // [rsp+38h] [rbp-128h]
  _QWORD *v74; // [rsp+38h] [rbp-128h]
  _QWORD *v75; // [rsp+38h] [rbp-128h]
  _QWORD *v76; // [rsp+38h] [rbp-128h]
  _QWORD *v77; // [rsp+38h] [rbp-128h]
  _QWORD *v78; // [rsp+38h] [rbp-128h]
  unsigned __int64 v80; // [rsp+40h] [rbp-120h]
  _QWORD *v81; // [rsp+40h] [rbp-120h]
  _QWORD *v82; // [rsp+40h] [rbp-120h]
  __int64 v83; // [rsp+40h] [rbp-120h]
  __int64 v84; // [rsp+40h] [rbp-120h]
  __int64 v85; // [rsp+40h] [rbp-120h]
  __int64 v86; // [rsp+40h] [rbp-120h]
  __int64 v87; // [rsp+40h] [rbp-120h]
  int v88; // [rsp+40h] [rbp-120h]
  __int64 v89; // [rsp+40h] [rbp-120h]
  const char *v90; // [rsp+40h] [rbp-120h]
  char v91; // [rsp+48h] [rbp-118h]
  int v93; // [rsp+68h] [rbp-F8h]
  const char *v94; // [rsp+70h] [rbp-F0h] BYREF
  _QWORD v95[2]; // [rsp+78h] [rbp-E8h] BYREF
  __int64 v96; // [rsp+88h] [rbp-D8h]
  __int64 v97; // [rsp+90h] [rbp-D0h]
  unsigned int *v98; // [rsp+A0h] [rbp-C0h] BYREF
  __int64 v99; // [rsp+A8h] [rbp-B8h]
  _BYTE v100[32]; // [rsp+B0h] [rbp-B0h] BYREF
  __int64 v101; // [rsp+D0h] [rbp-90h]
  __int64 v102; // [rsp+D8h] [rbp-88h]
  __int16 v103; // [rsp+E0h] [rbp-80h]
  __int64 v104; // [rsp+E8h] [rbp-78h]
  void **v105; // [rsp+F0h] [rbp-70h]
  void **v106; // [rsp+F8h] [rbp-68h]
  __int64 v107; // [rsp+100h] [rbp-60h]
  int v108; // [rsp+108h] [rbp-58h]
  __int16 v109; // [rsp+10Ch] [rbp-54h]
  char v110; // [rsp+10Eh] [rbp-52h]
  __int64 v111; // [rsp+110h] [rbp-50h]
  __int64 v112; // [rsp+118h] [rbp-48h]
  void *v113; // [rsp+120h] [rbp-40h] BYREF
  void *v114; // [rsp+128h] [rbp-38h] BYREF

  v6 = (_QWORD *)*a2;
  if ( !a4 )
    BUG();
  v7 = *(_QWORD *)(a4 + 16);
  v70 = a4 - 24;
  v110 = 7;
  v104 = sub_AA48A0(v7);
  v105 = &v113;
  v106 = &v114;
  v101 = v7;
  v113 = &unk_49DA100;
  v98 = (unsigned int *)v100;
  v99 = 0x200000000LL;
  v114 = &unk_49DA0B0;
  v107 = 0;
  v108 = 0;
  v109 = 512;
  v111 = 0;
  v112 = 0;
  v102 = a4;
  v103 = a5;
  if ( a4 != v7 + 48 )
  {
    v11 = *(const char **)sub_B46C60(v70);
    v94 = v11;
    if ( v11 && (sub_B96E90((__int64)&v94, (__int64)v11, 1), (v13 = v94) != 0) )
    {
      v14 = v98;
      v15 = v99;
      v16 = &v98[4 * (unsigned int)v99];
      if ( v98 != v16 )
      {
        while ( 1 )
        {
          v12 = *v14;
          if ( !(_DWORD)v12 )
            break;
          v14 += 4;
          if ( v16 == v14 )
            goto LABEL_22;
        }
        *((_QWORD *)v14 + 1) = v94;
        goto LABEL_10;
      }
LABEL_22:
      if ( (unsigned int)v99 >= (unsigned __int64)HIDWORD(v99) )
      {
        v59 = (unsigned int)v99 + 1LL;
        if ( HIDWORD(v99) < v59 )
        {
          v90 = v94;
          sub_C8D5F0((__int64)&v98, v100, v59, 0x10u, (__int64)v94, v12);
          v13 = v90;
          v16 = &v98[4 * (unsigned int)v99];
        }
        *(_QWORD *)v16 = 0;
        *((_QWORD *)v16 + 1) = v13;
        v13 = v94;
        LODWORD(v99) = v99 + 1;
      }
      else
      {
        if ( v16 )
        {
          *v16 = 0;
          *((_QWORD *)v16 + 1) = v13;
          v15 = v99;
          v13 = v94;
        }
        LODWORD(v99) = v15 + 1;
      }
    }
    else
    {
      sub_93FB40((__int64)&v98, 0);
      v13 = v94;
    }
    if ( !v13 )
      goto LABEL_13;
LABEL_10:
    sub_B91220((__int64)&v94, (__int64)v13);
  }
LABEL_13:
  v17 = *(_QWORD *)(a3 - 32LL * (*(_DWORD *)(a3 + 4) & 0x7FFFFFF));
  v18 = *(unsigned int *)(a1 + 24);
  if ( (_DWORD)v18 )
  {
    v19 = *(_QWORD *)(a1 + 8);
    v20 = (v18 - 1) & (((unsigned int)v17 >> 9) ^ ((unsigned int)v17 >> 4));
    v21 = v19 + 48LL * v20;
    v22 = *(_QWORD *)(v21 + 24);
    if ( v17 == v22 )
    {
LABEL_15:
      if ( v21 != v19 + 48 * v18 )
      {
        v23 = *(_QWORD **)(v21 + 40);
        goto LABEL_17;
      }
    }
    else
    {
      v33 = 1;
      while ( v22 != -4096 )
      {
        v52 = v33 + 1;
        v20 = (v18 - 1) & (v33 + v20);
        v21 = v19 + 48LL * v20;
        v22 = *(_QWORD *)(v21 + 24);
        if ( v17 == v22 )
          goto LABEL_15;
        v33 = v52;
      }
    }
  }
  v72 = *(_QWORD *)(a3 - 32LL * (*(_DWORD *)(a3 + 4) & 0x7FFFFFF));
  v67 = *(_BYTE *)(v17 + 80) & 1;
  v91 = *(_BYTE *)(v17 + 32) & 0xF;
  v68 = *(_QWORD *)(v17 - 32);
  v81 = *(_QWORD **)(v17 + 24);
  v94 = sub_BD5D20(v72);
  LOWORD(v97) = 261;
  v95[0] = v34;
  v35 = *(_BYTE *)(v72 + 33);
  v93 = 1;
  v36 = sub_BD2C40(88, unk_3F0FAE8);
  v17 = v72;
  v23 = v36;
  v37 = (v35 >> 2) & 7;
  if ( v36 )
  {
    v38 = v81;
    v82 = v36;
    sub_B30000((__int64)v36, (__int64)a2, v38, v67, v91, v68, (__int64)&v94, v72, v37, 0x100000001LL, 0);
    v23 = v82;
    v17 = v72;
  }
  v95[0] = 2;
  v95[1] = 0;
  v96 = v17;
  if ( v17 != -8192 && v17 != -4096 )
  {
    v73 = v23;
    v83 = v17;
    sub_BD73F0((__int64)v95);
    v23 = v73;
    v17 = v83;
  }
  v39 = *(_DWORD *)(a1 + 24);
  v97 = a1;
  v94 = (const char *)&unk_4A259B8;
  if ( !v39 )
  {
    ++*(_QWORD *)a1;
    goto LABEL_36;
  }
  v41 = v96;
  v48 = *(_QWORD *)(a1 + 8);
  v49 = (v39 - 1) & (((unsigned int)v96 >> 4) ^ ((unsigned int)v96 >> 9));
  v50 = (unsigned __int64 *)(v48 + 48LL * v49);
  v51 = v50[3];
  if ( v51 != v96 )
  {
    v88 = 1;
    v42 = 0;
    while ( v51 != -4096 )
    {
      if ( !v42 && v51 == -8192 )
        v42 = v50;
      v49 = (v39 - 1) & (v88 + v49);
      v50 = (unsigned __int64 *)(v48 + 48LL * v49);
      v51 = v50[3];
      if ( v96 == v51 )
        goto LABEL_49;
      ++v88;
    }
    v60 = *(_DWORD *)(a1 + 16);
    if ( !v42 )
      v42 = v50;
    ++*(_QWORD *)a1;
    v43 = v60 + 1;
    if ( 4 * (v60 + 1) < 3 * v39 )
    {
      if ( v39 - *(_DWORD *)(a1 + 20) - v43 > v39 >> 3 )
      {
LABEL_39:
        *(_DWORD *)(a1 + 16) = v43;
        if ( v42[3] == -4096 )
        {
          v45 = v42 + 1;
          if ( v41 != -4096 )
          {
LABEL_44:
            v42[3] = v41;
            if ( v41 == 0 || v41 == -4096 || v41 == -8192 )
            {
              v41 = v96;
            }
            else
            {
              v76 = v23;
              v86 = v17;
              sub_BD6050(v45, v95[0] & 0xFFFFFFFFFFFFFFF8LL);
              v41 = v96;
              v23 = v76;
              v17 = v86;
            }
          }
        }
        else
        {
          --*(_DWORD *)(a1 + 20);
          v44 = v42[3];
          if ( v44 != v41 )
          {
            v45 = v42 + 1;
            if ( v44 != 0 && v44 != -4096 && v44 != -8192 )
            {
              v75 = v23;
              v85 = v17;
              sub_BD60C0(v42 + 1);
              v41 = v96;
              v23 = v75;
              v17 = v85;
            }
            goto LABEL_44;
          }
        }
        v46 = v97;
        v47 = v42 + 5;
        *v47 = 0;
        *(v47 - 1) = v46;
        goto LABEL_50;
      }
      v78 = v23;
      v89 = v17;
      sub_2CF9C20(a1, v39);
      v61 = *(_DWORD *)(a1 + 24);
      v17 = v89;
      v23 = v78;
      if ( !v61 )
        goto LABEL_37;
      v41 = v96;
      v62 = v61 - 1;
      v63 = *(_QWORD *)(a1 + 8);
      v64 = 1;
      v65 = (v61 - 1) & (((unsigned int)v96 >> 9) ^ ((unsigned int)v96 >> 4));
      v57 = 0;
      v42 = (unsigned __int64 *)(v63 + 48LL * v65);
      v66 = v42[3];
      if ( v66 == v96 )
        goto LABEL_38;
      while ( v66 != -4096 )
      {
        if ( !v57 && v66 == -8192 )
          v57 = v42;
        v65 = v62 & (v64 + v65);
        v42 = (unsigned __int64 *)(v63 + 48LL * v65);
        v66 = v42[3];
        if ( v96 == v66 )
          goto LABEL_38;
        ++v64;
      }
      goto LABEL_58;
    }
LABEL_36:
    v74 = v23;
    v84 = v17;
    sub_2CF9C20(a1, 2 * v39);
    v40 = *(_DWORD *)(a1 + 24);
    v17 = v84;
    v23 = v74;
    if ( !v40 )
    {
LABEL_37:
      v41 = v96;
      v42 = 0;
LABEL_38:
      v43 = *(_DWORD *)(a1 + 16) + 1;
      goto LABEL_39;
    }
    v41 = v96;
    v53 = v40 - 1;
    v54 = *(_QWORD *)(a1 + 8);
    v55 = 1;
    v56 = (v40 - 1) & (((unsigned int)v96 >> 9) ^ ((unsigned int)v96 >> 4));
    v57 = 0;
    v42 = (unsigned __int64 *)(v54 + 48LL * v56);
    v58 = v42[3];
    if ( v96 == v58 )
      goto LABEL_38;
    while ( v58 != -4096 )
    {
      if ( v58 == -8192 && !v57 )
        v57 = v42;
      v56 = v53 & (v55 + v56);
      v42 = (unsigned __int64 *)(v54 + 48LL * v56);
      v58 = v42[3];
      if ( v96 == v58 )
        goto LABEL_38;
      ++v55;
    }
LABEL_58:
    if ( v57 )
      v42 = v57;
    goto LABEL_38;
  }
LABEL_49:
  v47 = v50 + 5;
LABEL_50:
  v94 = (const char *)&unk_49DB368;
  if ( v41 != -4096 && v41 != 0 && v41 != -8192 )
  {
    v77 = v23;
    v87 = v17;
    sub_BD60C0(v95);
    v23 = v77;
    v17 = v87;
  }
  *v47 = v23;
LABEL_17:
  v71 = v17;
  v80 = (unsigned __int64)v23;
  v24 = *(_DWORD *)(v23[1] + 8LL);
  v25 = (__int64 **)sub_BCB2B0(v6);
  v26 = (__int64 **)sub_BCE760(v25, v24 >> 8);
  v94 = "bcast";
  LOWORD(v97) = 259;
  v27 = sub_2CF9670((__int64 *)&v98, 0x31u, v80, v26, (__int64)&v94, 0, v93, 0);
  v28 = (__int64 **)sub_BCB2B0(v6);
  v29 = (__int64 **)sub_BCE760(v28, 0);
  LOWORD(v97) = 257;
  v30 = sub_2CF9670((__int64 *)&v98, 0x32u, v27, v29, (__int64)&v94, 0, v93, 0);
  if ( !a6 )
  {
    v32 = (__int64 **)sub_BCE760(*(__int64 ***)(v71 + 24), 0);
    v94 = "bcast";
    LOWORD(v97) = 259;
    v30 = sub_2CF9670((__int64 *)&v98, 0x31u, v30, v32, (__int64)&v94, 0, v93, 0);
  }
  nullsub_61();
  v113 = &unk_49DA100;
  nullsub_63();
  if ( v98 != (unsigned int *)v100 )
    _libc_free((unsigned __int64)v98);
  return v30;
}
