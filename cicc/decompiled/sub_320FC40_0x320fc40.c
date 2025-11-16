// Function: sub_320FC40
// Address: 0x320fc40
//
void __fastcall sub_320FC40(__int64 a1, __int64 a2)
{
  __int64 v3; // r14
  __int64 v4; // rbx
  __int64 v5; // r13
  __int64 v6; // r12
  _QWORD *v7; // rax
  __int64 v8; // rcx
  __int64 v9; // r9
  __int64 v10; // rax
  int v11; // edx
  __int64 v12; // rax
  char v13; // al
  __int64 v14; // rbx
  __int64 (*v15)(); // rax
  int v16; // ebx
  char v17; // r9
  int v18; // eax
  int v19; // ebx
  __int64 v20; // r13
  __int64 v21; // r14
  __int64 v22; // r9
  char v23; // r12
  __int64 v24; // rbx
  __int64 v25; // rdx
  __int64 v26; // rsi
  __int64 v27; // r14
  __int64 v28; // rbx
  __int64 v29; // r13
  __int64 v30; // rax
  unsigned __int8 *v31; // rdx
  unsigned int v32; // esi
  __int64 v33; // r8
  unsigned int v34; // edi
  _QWORD *v35; // rax
  __int64 v36; // rcx
  unsigned int v37; // esi
  unsigned int v38; // edi
  _QWORD *v39; // rax
  __int64 v40; // rcx
  bool v41; // zf
  int v42; // r11d
  _QWORD *v43; // rdx
  int v44; // eax
  int v45; // ecx
  int v46; // r10d
  _QWORD *v47; // rdx
  int v48; // eax
  int v49; // ecx
  int v50; // r9d
  int v51; // r9d
  __int64 v52; // r10
  __int64 v53; // rax
  __int64 v54; // r8
  int v55; // edi
  _QWORD *v56; // rsi
  int v57; // edi
  int v58; // edi
  __int64 v59; // r9
  __int64 v60; // rax
  __int64 v61; // r8
  int v62; // esi
  _QWORD *v63; // r10
  int v64; // edi
  int v65; // edi
  __int64 v66; // r8
  _QWORD *v67; // r9
  __int64 v68; // r12
  int v69; // eax
  __int64 v70; // rsi
  int v71; // edi
  int v72; // edi
  __int64 v73; // r8
  _QWORD *v74; // r9
  __int64 v75; // r12
  int v76; // eax
  __int64 v77; // rsi
  char v78; // r9
  int v79; // eax
  __int64 v80; // rax
  int v81; // eax
  int v82; // edx
  int v83; // edx
  __int64 v84; // [rsp+8h] [rbp-88h]
  __int64 v85; // [rsp+10h] [rbp-80h]
  __int64 v86; // [rsp+10h] [rbp-80h]
  unsigned __int64 v87; // [rsp+20h] [rbp-70h]
  __int64 v88; // [rsp+20h] [rbp-70h]
  __int64 v89; // [rsp+20h] [rbp-70h]
  __int64 v91; // [rsp+38h] [rbp-58h] BYREF
  __int64 v92; // [rsp+40h] [rbp-50h] BYREF
  _QWORD *v93; // [rsp+48h] [rbp-48h] BYREF
  __int64 (__fastcall *v94)(__int64 *, __int64 *, int); // [rsp+50h] [rbp-40h]
  unsigned __int64 (__fastcall *v95)(__int64 *, __int64, __int64); // [rsp+58h] [rbp-38h]

  v3 = *(_QWORD *)(a2 + 16);
  v4 = (*(__int64 (__fastcall **)(__int64))(*(_QWORD *)v3 + 200LL))(v3);
  v5 = *(_QWORD *)(a2 + 48);
  v6 = *(_QWORD *)a2;
  v7 = (_QWORD *)sub_22077B0(0x1F0u);
  if ( v7 )
  {
    memset(v7, 0, 0x1F0u);
    v7[1] = 1;
    v7[10] = v7 + 12;
    v7[16] = v7 + 14;
    v7[17] = v7 + 14;
    v7[19] = v7 + 21;
    *v7 = v7 + 6;
    v7[32] = v7 + 34;
    v7[7] = v7 + 9;
    v7[36] = v7 + 42;
    v8 = (__int64)(v7 + 45);
    v7[8] = 0x100000000LL;
    v7[11] = 0x100000000LL;
    v7[20] = 0x100000000LL;
    v7[33] = 0x100000000LL;
    v7[37] = 1;
    v7[43] = v7 + 45;
    v7[44] = 0x100000000LL;
    *((_DWORD *)v7 + 8) = 1065353216;
    *((_DWORD *)v7 + 80) = 1065353216;
  }
  v93 = v7;
  v92 = v6;
  v10 = sub_320F8B0(a1 + 1056, &v92, (__int64 *)&v93, v8, a1 + 1056, v9);
  if ( v93 )
  {
    v85 = v10;
    v87 = (unsigned __int64)v93;
    sub_31FB410((__int64)v93);
    j_j___libc_free_0(v87);
    v10 = v85;
  }
  v11 = *(_DWORD *)(a1 + 1048);
  v12 = *(_QWORD *)(v10 + 8);
  *(_QWORD *)(a1 + 792) = v12;
  *(_DWORD *)(a1 + 1048) = v11 + 1;
  *(_DWORD *)(v12 + 456) = v11;
  *(_QWORD *)(*(_QWORD *)(a1 + 792) + 440LL) = *(_QWORD *)(*(_QWORD *)(a1 + 8) + 536LL);
  *(_DWORD *)(*(_QWORD *)(a1 + 792) + 472LL) = *(_DWORD *)(v5 + 88);
  *(_DWORD *)(*(_QWORD *)(a1 + 792) + 464LL) = *(_QWORD *)(v5 + 48);
  *(_DWORD *)(*(_QWORD *)(a1 + 792) + 476LL) = *(_QWORD *)(v5 + 56);
  v88 = *(_QWORD *)(a1 + 792);
  v13 = (*(__int64 (__fastcall **)(__int64, __int64))(*(_QWORD *)v4 + 544LL))(v4, a2);
  if ( v13 )
    v13 = (*(__int64 (__fastcall **)(__int64, __int64))(*(_QWORD *)v4 + 536LL))(v4, a2);
  *(_BYTE *)(v88 + 488) = v13;
  *(_BYTE *)(*(_QWORD *)(a1 + 792) + 481LL) = 0;
  *(_BYTE *)(*(_QWORD *)(a1 + 792) + 480LL) = 0;
  if ( *(_DWORD *)(*(_QWORD *)(a1 + 792) + 464LL) )
  {
    v14 = 0;
    v15 = *(__int64 (**)())(*(_QWORD *)v3 + 136LL);
    if ( v15 != sub_2DD19D0 )
      v14 = ((__int64 (__fastcall *)(__int64))v15)(v3);
    if ( (unsigned __int8)sub_B2D610(*(_QWORD *)a2, 20)
      || !(*(unsigned __int8 (__fastcall **)(__int64, __int64))(*(_QWORD *)v14 + 392LL))(v14, a2) )
    {
      *(_BYTE *)(*(_QWORD *)(a1 + 792) + 480LL) = 1;
      *(_BYTE *)(*(_QWORD *)(a1 + 792) + 481LL) = 1;
    }
    else
    {
      *(_BYTE *)(*(_QWORD *)(a1 + 792) + 490LL) = 1;
      *(_BYTE *)(*(_QWORD *)(a1 + 792) + 481LL) = 2;
      *(_BYTE *)(*(_QWORD *)(a1 + 792) + 480LL) = (*(_BYTE *)(*(_QWORD *)(a1 + 792) + 488LL) == 0) + 1;
    }
  }
  v16 = *(unsigned __int8 *)(v5 + 36);
  if ( *(_BYTE *)(a2 + 341) )
    v16 |= 2u;
  if ( *(_BYTE *)(a2 + 342) )
    v16 |= 8u;
  if ( (*(_BYTE *)(v6 + 2) & 8) != 0 )
  {
    v80 = sub_B2E500(v6);
    v81 = sub_B2A630(v80);
    v82 = v16;
    v16 |= 0x10u;
    v83 = v82 | 0x40;
    if ( (unsigned int)(v81 - 7) <= 1 )
      v16 = v83;
  }
  if ( (unsigned __int8)sub_B2D610(v6, 16) )
    v16 |= 0x20u;
  v17 = sub_B2D610(v6, 20);
  v18 = v16;
  if ( v17 )
  {
    LOBYTE(v18) = v16 | 0x80;
    v16 = v18;
  }
  if ( *(_DWORD *)(v5 + 68) == -1 )
  {
    v78 = sub_B2DC10(v6);
    v79 = v16;
    if ( !v78 )
    {
      BYTE1(v79) = BYTE1(v16) | 0x20;
      v16 = v79;
    }
  }
  else if ( (unsigned __int8)sub_B2D610(v6, 71) || (unsigned __int8)sub_B2D610(v6, 70) )
  {
    BYTE1(v16) |= 0x11u;
  }
  else
  {
    BYTE1(v16) |= 1u;
  }
  v19 = (*(unsigned __int8 *)(*(_QWORD *)(a1 + 792) + 481LL) << 16)
      | (*(unsigned __int8 *)(*(_QWORD *)(a1 + 792) + 480LL) << 14)
      | v16;
  if ( *(_DWORD *)(*(_QWORD *)(*(_QWORD *)(a1 + 8) + 200LL) + 648LL)
    && !(unsigned __int8)sub_B2D610(v6, 47)
    && !(unsigned __int8)sub_B2D610(v6, 18)
    && !(unsigned __int8)sub_B2D610(v6, 48) )
  {
    v19 |= 0x100000u;
  }
  sub_B2EE70((__int64)&v92, v6, 0);
  if ( (_BYTE)v94 )
    v19 |= 0xC0000u;
  *(_DWORD *)(*(_QWORD *)(a1 + 792) + 484LL) = v19;
  (*(void (__fastcall **)(_QWORD, _QWORD))(**(_QWORD **)(a1 + 528) + 720LL))(
    *(_QWORD *)(a1 + 528),
    *(unsigned int *)(*(_QWORD *)(a1 + 792) + 456LL));
  v91 = 0;
  v20 = *(_QWORD *)(a2 + 328);
  v21 = a2 + 320;
  v89 = a2 + 320;
  if ( a2 + 320 == v20 )
    goto LABEL_58;
  v22 = 0;
  v23 = 1;
  do
  {
LABEL_30:
    v24 = *(_QWORD *)(v20 + 56);
    v25 = v20 + 48;
    if ( v20 + 48 == v24 )
      goto LABEL_39;
    while ( (*(_BYTE *)(*(_QWORD *)(v24 + 16) + 24LL) & 0x10) != 0 )
    {
LABEL_37:
      if ( (*(_BYTE *)v24 & 4) != 0 )
      {
        v24 = *(_QWORD *)(v24 + 8);
        if ( v25 == v24 )
          goto LABEL_39;
      }
      else
      {
        while ( (*(_BYTE *)(v24 + 44) & 8) != 0 )
          v24 = *(_QWORD *)(v24 + 8);
        v24 = *(_QWORD *)(v24 + 8);
        if ( v25 == v24 )
          goto LABEL_39;
      }
    }
    if ( (*(_BYTE *)(v24 + 44) & 1) != 0 || (v26 = *(_QWORD *)(v24 + 56)) == 0 )
    {
      v23 = 0;
      goto LABEL_37;
    }
    if ( &v91 == (__int64 *)(v24 + 56) )
      goto LABEL_39;
    if ( !v22 )
    {
      v91 = *(_QWORD *)(v24 + 56);
LABEL_120:
      sub_B96E90((__int64)&v91, v26, 1);
      v20 = *(_QWORD *)(v20 + 8);
      v22 = v91;
      if ( v21 == v20 )
        break;
      goto LABEL_30;
    }
    sub_B91220((__int64)&v91, v22);
    v91 = *(_QWORD *)(v24 + 56);
    v22 = v91;
    if ( v91 )
    {
      v26 = v91;
      goto LABEL_120;
    }
LABEL_39:
    v20 = *(_QWORD *)(v20 + 8);
  }
  while ( v21 != v20 );
  if ( v22 )
  {
    if ( !v23 )
    {
      sub_B10E30(&v92, &v91);
      sub_320DBD0((_QWORD *)a1, &v92);
      if ( v92 )
        sub_B91220((__int64)&v92, v92);
    }
  }
  if ( v89 != *(_QWORD *)(a2 + 328) )
  {
    v27 = *(_QWORD *)(a2 + 328);
    v86 = a1 + 432;
    while ( 1 )
    {
      v28 = *(_QWORD *)(v27 + 56);
      v29 = v27 + 48;
      if ( v27 + 48 != v28 )
        break;
LABEL_57:
      v27 = *(_QWORD *)(v27 + 8);
      if ( v89 == v27 )
        goto LABEL_58;
    }
    while ( 2 )
    {
      v30 = *(_QWORD *)(v28 + 48);
      v31 = (unsigned __int8 *)(v30 & 0xFFFFFFFFFFFFFFF8LL);
      if ( (v30 & 0xFFFFFFFFFFFFFFF8LL) == 0
        || (v30 & 7) != 3
        || !v31[6]
        || !*(_QWORD *)&v31[8 * *(int *)v31 + 16 + 8 * (__int64)(v31[5] + v31[4])] )
      {
        goto LABEL_55;
      }
      v32 = *(_DWORD *)(a1 + 456);
      if ( v32 )
      {
        v33 = *(_QWORD *)(a1 + 440);
        v34 = (v32 - 1) & (((unsigned int)v28 >> 9) ^ ((unsigned int)v28 >> 4));
        v35 = (_QWORD *)(v33 + 16LL * v34);
        v36 = *v35;
        if ( *v35 == v28 )
          goto LABEL_53;
        v46 = 1;
        v47 = 0;
        while ( v36 != -4096 )
        {
          if ( v36 != -8192 || v47 )
            v35 = v47;
          v34 = (v32 - 1) & (v46 + v34);
          v36 = *(_QWORD *)(v33 + 16LL * v34);
          if ( v36 == v28 )
            goto LABEL_53;
          ++v46;
          v47 = v35;
          v35 = (_QWORD *)(v33 + 16LL * v34);
        }
        if ( !v47 )
          v47 = v35;
        v48 = *(_DWORD *)(a1 + 448);
        ++*(_QWORD *)(a1 + 432);
        v49 = v48 + 1;
        if ( 4 * (v48 + 1) < 3 * v32 )
        {
          if ( v32 - *(_DWORD *)(a1 + 452) - v49 <= v32 >> 3 )
          {
            sub_31FF3C0(v86, v32);
            v71 = *(_DWORD *)(a1 + 456);
            if ( !v71 )
            {
LABEL_168:
              ++*(_DWORD *)(a1 + 448);
              BUG();
            }
            v72 = v71 - 1;
            v73 = *(_QWORD *)(a1 + 440);
            v74 = 0;
            LODWORD(v75) = v72 & (((unsigned int)v28 >> 9) ^ ((unsigned int)v28 >> 4));
            v49 = *(_DWORD *)(a1 + 448) + 1;
            v76 = 1;
            v47 = (_QWORD *)(v73 + 16LL * (unsigned int)v75);
            v77 = *v47;
            if ( *v47 != v28 )
            {
              while ( v77 != -4096 )
              {
                if ( v77 == -8192 && !v74 )
                  v74 = v47;
                v75 = v72 & (unsigned int)(v75 + v76);
                v47 = (_QWORD *)(v73 + 16 * v75);
                v77 = *v47;
                if ( *v47 == v28 )
                  goto LABEL_84;
                ++v76;
              }
              if ( v74 )
                v47 = v74;
            }
          }
          goto LABEL_84;
        }
      }
      else
      {
        ++*(_QWORD *)(a1 + 432);
      }
      sub_31FF3C0(v86, 2 * v32);
      v50 = *(_DWORD *)(a1 + 456);
      if ( !v50 )
        goto LABEL_168;
      v51 = v50 - 1;
      v52 = *(_QWORD *)(a1 + 440);
      LODWORD(v53) = v51 & (((unsigned int)v28 >> 9) ^ ((unsigned int)v28 >> 4));
      v49 = *(_DWORD *)(a1 + 448) + 1;
      v47 = (_QWORD *)(v52 + 16LL * (unsigned int)v53);
      v54 = *v47;
      if ( *v47 != v28 )
      {
        v55 = 1;
        v56 = 0;
        while ( v54 != -4096 )
        {
          if ( v54 == -8192 && !v56 )
            v56 = v47;
          v53 = v51 & (unsigned int)(v53 + v55);
          v47 = (_QWORD *)(v52 + 16 * v53);
          v54 = *v47;
          if ( *v47 == v28 )
            goto LABEL_84;
          ++v55;
        }
        if ( v56 )
          v47 = v56;
      }
LABEL_84:
      *(_DWORD *)(a1 + 448) = v49;
      if ( *v47 != -4096 )
        --*(_DWORD *)(a1 + 452);
      *v47 = v28;
      v47[1] = 0;
LABEL_53:
      v37 = *(_DWORD *)(a1 + 488);
      if ( v37 )
      {
        v38 = (v37 - 1) & (((unsigned int)v28 >> 9) ^ ((unsigned int)v28 >> 4));
        v39 = (_QWORD *)(*(_QWORD *)(a1 + 472) + 16LL * v38);
        v40 = *v39;
        if ( *v39 == v28 )
          goto LABEL_55;
        v84 = *(_QWORD *)(a1 + 472);
        v42 = 1;
        v43 = 0;
        while ( v40 != -4096 )
        {
          if ( v40 != -8192 || v43 )
            v39 = v43;
          v38 = (v37 - 1) & (v42 + v38);
          v40 = *(_QWORD *)(v84 + 16LL * v38);
          if ( v40 == v28 )
            goto LABEL_55;
          ++v42;
          v43 = v39;
          v39 = (_QWORD *)(v84 + 16LL * v38);
        }
        if ( !v43 )
          v43 = v39;
        v44 = *(_DWORD *)(a1 + 480);
        ++*(_QWORD *)(a1 + 464);
        v45 = v44 + 1;
        if ( 4 * (v44 + 1) < 3 * v37 )
        {
          if ( v37 - *(_DWORD *)(a1 + 484) - v45 <= v37 >> 3 )
          {
            sub_31FF3C0(a1 + 464, v37);
            v64 = *(_DWORD *)(a1 + 488);
            if ( !v64 )
            {
LABEL_169:
              ++*(_DWORD *)(a1 + 480);
              BUG();
            }
            v65 = v64 - 1;
            v66 = *(_QWORD *)(a1 + 472);
            v67 = 0;
            LODWORD(v68) = v65 & (((unsigned int)v28 >> 9) ^ ((unsigned int)v28 >> 4));
            v45 = *(_DWORD *)(a1 + 480) + 1;
            v69 = 1;
            v43 = (_QWORD *)(v66 + 16LL * (unsigned int)v68);
            v70 = *v43;
            if ( *v43 != v28 )
            {
              while ( v70 != -4096 )
              {
                if ( !v67 && v70 == -8192 )
                  v67 = v43;
                v68 = v65 & (unsigned int)(v68 + v69);
                v43 = (_QWORD *)(v66 + 16 * v68);
                v70 = *v43;
                if ( *v43 == v28 )
                  goto LABEL_75;
                ++v69;
              }
              if ( v67 )
                v43 = v67;
            }
          }
LABEL_75:
          *(_DWORD *)(a1 + 480) = v45;
          if ( *v43 != -4096 )
            --*(_DWORD *)(a1 + 484);
          *v43 = v28;
          v43[1] = 0;
LABEL_55:
          if ( (*(_BYTE *)v28 & 4) != 0 )
          {
            v28 = *(_QWORD *)(v28 + 8);
            if ( v29 == v28 )
              goto LABEL_57;
          }
          else
          {
            while ( (*(_BYTE *)(v28 + 44) & 8) != 0 )
              v28 = *(_QWORD *)(v28 + 8);
            v28 = *(_QWORD *)(v28 + 8);
            if ( v29 == v28 )
              goto LABEL_57;
          }
          continue;
        }
      }
      else
      {
        ++*(_QWORD *)(a1 + 464);
      }
      break;
    }
    sub_31FF3C0(a1 + 464, 2 * v37);
    v57 = *(_DWORD *)(a1 + 488);
    if ( !v57 )
      goto LABEL_169;
    v58 = v57 - 1;
    v59 = *(_QWORD *)(a1 + 472);
    LODWORD(v60) = v58 & (((unsigned int)v28 >> 9) ^ ((unsigned int)v28 >> 4));
    v45 = *(_DWORD *)(a1 + 480) + 1;
    v43 = (_QWORD *)(v59 + 16LL * (unsigned int)v60);
    v61 = *v43;
    if ( v28 != *v43 )
    {
      v62 = 1;
      v63 = 0;
      while ( v61 != -4096 )
      {
        if ( v61 == -8192 && !v63 )
          v63 = v43;
        v60 = v58 & (unsigned int)(v60 + v62);
        v43 = (_QWORD *)(v59 + 16 * v60);
        v61 = *v43;
        if ( *v43 == v28 )
          goto LABEL_75;
        ++v62;
      }
      if ( v63 )
        v43 = v63;
    }
    goto LABEL_75;
  }
LABEL_58:
  v41 = *(_DWORD *)(*(_QWORD *)(*(_QWORD *)(a1 + 16) + 2488LL) + 264LL) == 36;
  v92 = a1;
  v95 = sub_31FF5A0;
  v94 = sub_31F3D30;
  sub_31F99A0(a2, v41, (__int64)&v92);
  if ( v94 )
    v94(&v92, &v92, 3);
  if ( v91 )
    sub_B91220((__int64)&v91, v91);
}
