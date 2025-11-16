// Function: sub_2AF9E50
// Address: 0x2af9e50
//
__int64 *__fastcall sub_2AF9E50(__int64 a1, __int64 *a2, __int64 a3, __int64 *a4, __int64 a5, __int64 a6)
{
  __int64 *v8; // rbx
  unsigned __int64 v9; // rax
  char v10; // al
  __int64 *v11; // r14
  _QWORD *v12; // r13
  unsigned int v13; // ecx
  __int64 v14; // r11
  __int64 v15; // r14
  __int64 v16; // rbx
  __int64 v17; // rdx
  unsigned int v18; // esi
  char v19; // al
  char *v20; // rdx
  __int64 v21; // rdx
  char v22; // cl
  unsigned __int64 v23; // rdx
  unsigned __int64 v24; // rax
  __int64 v25; // rax
  __int64 v26; // rdx
  int v27; // eax
  __int64 v28; // r9
  __int64 v29; // rdx
  int v30; // r14d
  __int64 v31; // r13
  __int64 v32; // rax
  int v33; // r8d
  __int64 v34; // rax
  __int64 v35; // rax
  __int64 v36; // r9
  __int64 v37; // rdx
  int v38; // eax
  char v40; // cl
  unsigned __int64 v41; // rdx
  unsigned __int64 v42; // rax
  __int64 v43; // rax
  unsigned int v44; // edx
  __int64 v45; // r9
  __int64 v46; // r13
  __int64 v47; // r14
  __int64 v48; // r8
  __int64 v49; // rax
  unsigned __int64 v50; // rdx
  __int64 v51; // rax
  __int64 v52; // rax
  __int64 v53; // r9
  __int64 v54; // rdx
  char v55; // al
  __int64 v56; // rdx
  int v57; // eax
  char *v58; // rdx
  char v59; // cl
  unsigned __int64 v60; // rdx
  unsigned __int64 v61; // rax
  _QWORD *v62; // rax
  _BYTE *v63; // r14
  __int64 v64; // r13
  __int64 v65; // r12
  __int64 v66; // rdx
  unsigned int v67; // esi
  char v68; // al
  __int64 v69; // rdi
  _DWORD *v70; // r11
  __int64 v71; // r10
  __int64 v72; // r12
  __int64 (__fastcall *v73)(__int64, _BYTE *, _BYTE *, __int64, __int64); // rax
  __int64 v74; // rax
  __int64 v75; // r13
  __int64 *v76; // rdi
  __int64 *v77; // rax
  __int64 v78; // rsi
  int v79; // edx
  _QWORD *v80; // rax
  __int64 v81; // r12
  __int64 v82; // rbx
  __int64 v83; // r12
  __int64 v84; // rdx
  unsigned int v85; // esi
  char v86; // al
  __int64 v87; // rax
  __int64 v88; // rax
  __int64 v89; // [rsp+8h] [rbp-E8h]
  __int64 v90; // [rsp+8h] [rbp-E8h]
  __int64 v91; // [rsp+10h] [rbp-E0h]
  __int64 v92; // [rsp+18h] [rbp-D8h]
  char v93; // [rsp+20h] [rbp-D0h]
  __int64 v94; // [rsp+28h] [rbp-C8h]
  unsigned __int8 v95; // [rsp+28h] [rbp-C8h]
  __int64 v96; // [rsp+30h] [rbp-C0h]
  __int64 *v97; // [rsp+30h] [rbp-C0h]
  __int64 v98; // [rsp+30h] [rbp-C0h]
  __int64 v99; // [rsp+30h] [rbp-C0h]
  _DWORD *v100; // [rsp+30h] [rbp-C0h]
  __int64 v101; // [rsp+30h] [rbp-C0h]
  _DWORD *v102; // [rsp+30h] [rbp-C0h]
  __int64 v103; // [rsp+38h] [rbp-B8h]
  int v104; // [rsp+38h] [rbp-B8h]
  char *v106; // [rsp+48h] [rbp-A8h]
  __int64 v107; // [rsp+48h] [rbp-A8h]
  const void *v108; // [rsp+48h] [rbp-A8h]
  __int64 v109; // [rsp+48h] [rbp-A8h]
  __int64 v110; // [rsp+58h] [rbp-98h]
  char *v111; // [rsp+60h] [rbp-90h] BYREF
  __int64 v112; // [rsp+68h] [rbp-88h]
  char *v113; // [rsp+70h] [rbp-80h]
  __int16 v114; // [rsp+80h] [rbp-70h]
  unsigned __int64 v115; // [rsp+90h] [rbp-60h] BYREF
  __int64 v116; // [rsp+98h] [rbp-58h]
  __int16 v117; // [rsp+B0h] [rbp-40h]

  v8 = a2;
  v9 = *(unsigned __int8 *)(a3 + 8);
  v106 = (char *)a5;
  if ( (unsigned __int8)v9 > 3u && (_BYTE)v9 != 5 )
  {
    if ( (unsigned __int8)v9 > 0x14u )
      goto LABEL_85;
    v21 = 1463376;
    if ( !_bittest64(&v21, v9) )
    {
      if ( (_BYTE)v9 == 16 )
      {
        v22 = -1;
        v95 = *(_BYTE *)(a1 + 104);
        v23 = *(unsigned int *)(a1 + 108) | (unsigned __int64)(1LL << v95);
        v24 = v23 & -(__int64)v23;
        if ( v24 )
        {
          _BitScanReverse64(&v24, v24);
          v22 = 63 - (v24 ^ 0x3F);
        }
        *(_BYTE *)(a1 + 104) = v22;
        v25 = sub_9208B0(*(_QWORD *)a1, *(_QWORD *)(a3 + 24));
        v116 = v26;
        v115 = (unsigned __int64)(v25 + 7) >> 3;
        v27 = sub_CA1930(&v115);
        v29 = *(_QWORD *)(a3 + 32);
        v104 = v27;
        if ( (_DWORD)v29 )
        {
          v30 = 0;
          v31 = 0;
          v98 = (unsigned int)v29;
          v32 = *(unsigned int *)(a1 + 16);
          do
          {
            v33 = v31;
            if ( v32 + 1 > (unsigned __int64)*(unsigned int *)(a1 + 20) )
            {
              sub_C8D5F0(a1 + 8, (const void *)(a1 + 24), v32 + 1, 4u, (unsigned int)v31, v28);
              v32 = *(unsigned int *)(a1 + 16);
              v33 = v31;
            }
            *(_DWORD *)(*(_QWORD *)(a1 + 8) + 4 * v32) = v33;
            ++*(_DWORD *)(a1 + 16);
            v34 = sub_BCB2D0((_QWORD *)a2[9]);
            v35 = sub_ACD640(v34, v31, 0);
            v37 = *(unsigned int *)(a1 + 48);
            if ( v37 + 1 > (unsigned __int64)*(unsigned int *)(a1 + 52) )
            {
              v89 = v35;
              sub_C8D5F0(a1 + 40, (const void *)(a1 + 56), v37 + 1, 8u, v37 + 1, v36);
              v37 = *(unsigned int *)(a1 + 48);
              v35 = v89;
            }
            ++v31;
            *(_QWORD *)(*(_QWORD *)(a1 + 40) + 8 * v37) = v35;
            ++*(_DWORD *)(a1 + 48);
            *(_DWORD *)(a1 + 108) = v30;
            sub_2AF9E50(a1, a2, *(_QWORD *)(a3 + 24), a4, v106);
            v38 = *(_DWORD *)(a1 + 16);
            --*(_DWORD *)(a1 + 48);
            v30 += v104;
            v32 = (unsigned int)(v38 - 1);
            *(_DWORD *)(a1 + 16) = v32;
          }
          while ( v98 != v31 );
        }
LABEL_29:
        *(_BYTE *)(a1 + 104) = v95;
        return (__int64 *)v95;
      }
      if ( (_BYTE)v9 == 15 )
      {
        v40 = -1;
        v95 = *(_BYTE *)(a1 + 104);
        v41 = *(unsigned int *)(a1 + 108) | (unsigned __int64)(1LL << v95);
        v42 = v41 & -(__int64)v41;
        if ( v42 )
        {
          _BitScanReverse64(&v42, v42);
          v40 = 63 - (v42 ^ 0x3F);
        }
        *(_BYTE *)(a1 + 104) = v40;
        v43 = sub_AE4AC0(*(_QWORD *)a1, a3);
        v44 = *(_DWORD *)(a3 + 12);
        v45 = v43;
        if ( v44 )
        {
          v46 = 0;
          v47 = v43 + 24;
          v48 = 0;
          v49 = *(unsigned int *)(a1 + 16);
          v99 = v44;
          v50 = v49 + 1;
          if ( v49 + 1 > (unsigned __int64)*(unsigned int *)(a1 + 20) )
            goto LABEL_39;
          while ( 1 )
          {
            *(_DWORD *)(*(_QWORD *)(a1 + 8) + 4 * v49) = v48;
            ++*(_DWORD *)(a1 + 16);
            v51 = sub_BCB2D0((_QWORD *)a2[9]);
            v52 = sub_ACD640(v51, v46, 0);
            v54 = *(unsigned int *)(a1 + 48);
            if ( v54 + 1 > (unsigned __int64)*(unsigned int *)(a1 + 52) )
            {
              v90 = v52;
              sub_C8D5F0(a1 + 40, (const void *)(a1 + 56), v54 + 1, 8u, v54 + 1, v53);
              v54 = *(unsigned int *)(a1 + 48);
              v52 = v90;
            }
            v47 += 16;
            *(_QWORD *)(*(_QWORD *)(a1 + 40) + 8 * v54) = v52;
            ++*(_DWORD *)(a1 + 48);
            v55 = *(_BYTE *)(v47 - 8);
            v115 = *(_QWORD *)(v47 - 16);
            LOBYTE(v116) = v55;
            *(_DWORD *)(a1 + 108) = sub_CA1930(&v115);
            v56 = *(_QWORD *)(*(_QWORD *)(a3 + 16) + 8 * v46++);
            sub_2AF9E50(a1, a2, v56, a4, v106);
            v57 = *(_DWORD *)(a1 + 16);
            --*(_DWORD *)(a1 + 48);
            v49 = (unsigned int)(v57 - 1);
            *(_DWORD *)(a1 + 16) = v49;
            if ( v99 == v46 )
              break;
            v50 = v49 + 1;
            v48 = (unsigned int)v46;
            if ( v49 + 1 > (unsigned __int64)*(unsigned int *)(a1 + 20) )
            {
LABEL_39:
              sub_C8D5F0(a1 + 8, (const void *)(a1 + 24), v50, 4u, v48, v45);
              v49 = *(unsigned int *)(a1 + 16);
              LODWORD(v48) = v46;
            }
          }
        }
        goto LABEL_29;
      }
LABEL_85:
      BUG();
    }
  }
  v117 = 257;
  v10 = *(_BYTE *)(a5 + 32);
  if ( v10 )
  {
    if ( v10 == 1 )
    {
      v111 = ".aggrsplit";
      v114 = 259;
    }
    else
    {
      if ( *(_BYTE *)(a5 + 33) == 1 )
      {
        a6 = *(_QWORD *)(a5 + 8);
        v58 = *(char **)a5;
      }
      else
      {
        v58 = (char *)a5;
        v10 = 2;
      }
      v111 = v58;
      v112 = a6;
      v113 = ".aggrsplit";
      LOBYTE(v114) = v10;
      HIBYTE(v114) = 3;
    }
  }
  else
  {
    v114 = 256;
  }
  v11 = *(__int64 **)(a1 + 40);
  v103 = *(unsigned int *)(a1 + 48);
  v96 = *(_QWORD *)(a1 + 88);
  v94 = *(_QWORD *)(a1 + 96);
  v12 = sub_BD2C40(88, (int)v103 + 1);
  if ( v12 )
  {
    v13 = (v103 + 1) & 0x7FFFFFF;
    v14 = *(_QWORD *)(v96 + 8);
    if ( (unsigned int)*(unsigned __int8 *)(v14 + 8) - 17 > 1 )
    {
      v76 = &v11[v103];
      if ( v11 != v76 )
      {
        v77 = v11;
        while ( 1 )
        {
          v78 = *(_QWORD *)(*v77 + 8);
          v79 = *(unsigned __int8 *)(v78 + 8);
          if ( v79 == 17 )
          {
            v86 = 0;
            goto LABEL_78;
          }
          if ( v79 == 18 )
            break;
          if ( v76 == ++v77 )
            goto LABEL_6;
        }
        v86 = 1;
LABEL_78:
        BYTE4(v110) = v86;
        LODWORD(v110) = *(_DWORD *)(v78 + 32);
        v87 = sub_BCE1B0((__int64 *)v14, v110);
        v13 = (v103 + 1) & 0x7FFFFFF;
        v14 = v87;
      }
    }
LABEL_6:
    sub_B44260((__int64)v12, v14, 34, v13, 0, 0);
    v12[9] = v94;
    v12[10] = sub_B4DC50(v94, (__int64)v11, v103);
    sub_B4D9A0((__int64)v12, v96, v11, v103, (__int64)&v111);
  }
  sub_B4DDE0((__int64)v12, 3);
  (*(void (__fastcall **)(__int64, _QWORD *, unsigned __int64 *, __int64, __int64))(*(_QWORD *)v8[11] + 16LL))(
    v8[11],
    v12,
    &v115,
    v8[7],
    v8[8]);
  if ( *v8 != *v8 + 16LL * *((unsigned int *)v8 + 2) )
  {
    v97 = v8;
    v15 = *v8 + 16LL * *((unsigned int *)v8 + 2);
    v16 = *v8;
    do
    {
      v17 = *(_QWORD *)(v16 + 8);
      v18 = *(_DWORD *)v16;
      v16 += 16;
      sub_B99FD0((__int64)v12, v18, v17);
    }
    while ( v15 != v16 );
    v8 = v97;
  }
  v19 = v106[32];
  if ( v19 )
  {
    if ( v19 == 1 )
    {
      v111 = ".load";
      v114 = 259;
    }
    else
    {
      if ( v106[33] == 1 )
      {
        v20 = *(char **)v106;
        v92 = *((_QWORD *)v106 + 1);
      }
      else
      {
        v20 = v106;
        v19 = 2;
      }
      v111 = v20;
      LOBYTE(v114) = v19;
      v112 = v92;
      v113 = ".load";
      HIBYTE(v114) = 3;
    }
  }
  else
  {
    v114 = 256;
  }
  v59 = -1;
  v60 = *(unsigned int *)(a1 + 108) | (unsigned __int64)(1LL << *(_BYTE *)(a1 + 104));
  if ( (v60 & -(__int64)v60) != 0 )
  {
    _BitScanReverse64(&v61, v60 & -(__int64)v60);
    v59 = 63 - (v61 ^ 0x3F);
  }
  v117 = 257;
  v93 = v59;
  v62 = sub_BD2C40(80, 1u);
  v63 = v62;
  if ( v62 )
    sub_B4D190((__int64)v62, a3, (__int64)v12, (__int64)&v115, 0, v93, 0, 0);
  (*(void (__fastcall **)(__int64, _BYTE *, char **, __int64, __int64))(*(_QWORD *)v8[11] + 16LL))(
    v8[11],
    v63,
    &v111,
    v8[7],
    v8[8]);
  v64 = *v8;
  v65 = *v8 + 16LL * *((unsigned int *)v8 + 2);
  if ( *v8 != v65 )
  {
    do
    {
      v66 = *(_QWORD *)(v64 + 8);
      v67 = *(_DWORD *)v64;
      v64 += 16;
      sub_B99FD0((__int64)v63, v67, v66);
    }
    while ( v65 != v64 );
  }
  v68 = v106[32];
  if ( v68 )
  {
    if ( v68 == 1 )
    {
      v111 = ".aggrsplitinsert";
      v114 = 259;
    }
    else
    {
      if ( v106[33] == 1 )
      {
        v91 = *((_QWORD *)v106 + 1);
        v106 = *(char **)v106;
      }
      else
      {
        v68 = 2;
      }
      LOBYTE(v114) = v68;
      v113 = ".aggrsplitinsert";
      v111 = v106;
      HIBYTE(v114) = 3;
      v112 = v91;
    }
  }
  else
  {
    v114 = 256;
  }
  v69 = v8[10];
  v70 = *(_DWORD **)(a1 + 8);
  v71 = *(unsigned int *)(a1 + 16);
  v72 = *a4;
  v73 = *(__int64 (__fastcall **)(__int64, _BYTE *, _BYTE *, __int64, __int64))(*(_QWORD *)v69 + 88LL);
  if ( v73 == sub_9482E0 )
  {
    if ( *(_BYTE *)v72 > 0x15u || *v63 > 0x15u )
      goto LABEL_69;
    v100 = *(_DWORD **)(a1 + 8);
    v107 = *(unsigned int *)(a1 + 16);
    v74 = sub_AAAE30(*a4, (__int64)v63, v100, v107);
    v71 = v107;
    v70 = v100;
    v75 = v74;
  }
  else
  {
    v102 = *(_DWORD **)(a1 + 8);
    v109 = *(unsigned int *)(a1 + 16);
    v88 = v73(v69, (_BYTE *)v72, v63, (__int64)v70, v71);
    v70 = v102;
    v71 = v109;
    v75 = v88;
  }
  if ( !v75 )
  {
LABEL_69:
    v101 = v71;
    v117 = 257;
    v108 = v70;
    v80 = sub_BD2C40(104, unk_3F148BC);
    v75 = (__int64)v80;
    if ( v80 )
    {
      sub_B44260((__int64)v80, *(_QWORD *)(v72 + 8), 65, 2u, 0, 0);
      *(_QWORD *)(v75 + 72) = v75 + 88;
      *(_QWORD *)(v75 + 80) = 0x400000000LL;
      sub_B4FD20(v75, v72, (__int64)v63, v108, v101, (__int64)&v115);
    }
    (*(void (__fastcall **)(__int64, __int64, char **, __int64, __int64))(*(_QWORD *)v8[11] + 16LL))(
      v8[11],
      v75,
      &v111,
      v8[7],
      v8[8]);
    v81 = 16LL * *((unsigned int *)v8 + 2);
    v82 = *v8;
    v83 = v82 + v81;
    while ( v83 != v82 )
    {
      v84 = *(_QWORD *)(v82 + 8);
      v85 = *(_DWORD *)v82;
      v82 += 16;
      sub_B99FD0(v75, v85, v84);
    }
  }
  *a4 = v75;
  return a4;
}
