// Function: sub_2FFB300
// Address: 0x2ffb300
//
void __fastcall sub_2FFB300(__int64 a1, unsigned int a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  int v6; // r12d
  __int64 v8; // r14
  __int64 *v9; // rdx
  __int64 v10; // r15
  char v11; // al
  unsigned int v12; // eax
  unsigned __int64 v13; // rdx
  __int64 v14; // rcx
  unsigned int v15; // eax
  int v16; // r15d
  int v17; // r12d
  __int64 v18; // r10
  unsigned int v19; // edi
  int *v20; // rdx
  int v21; // ecx
  unsigned int v22; // esi
  __int64 v23; // rcx
  int v24; // r9d
  int v25; // r9d
  __int64 v26; // r10
  unsigned int v27; // ecx
  int v28; // r8d
  int v29; // eax
  int v30; // edi
  int *v31; // rsi
  __int64 v32; // r13
  unsigned __int64 v33; // rsi
  __int64 v34; // r14
  __int16 v35; // ax
  int v36; // eax
  __int64 v37; // rdi
  __int64 v38; // rax
  bool v39; // r15
  __int64 *v40; // rax
  char v41; // dl
  __int64 v42; // rdx
  __int64 v43; // rsi
  unsigned int v44; // ecx
  __int64 *v45; // rax
  unsigned int v46; // esi
  __int64 v47; // r10
  int v48; // r14d
  unsigned int v49; // ecx
  unsigned int *v50; // rdi
  _DWORD *v51; // rax
  int *v52; // rax
  __int64 v53; // rax
  unsigned int v54; // r12d
  unsigned __int64 v55; // rdx
  int v56; // eax
  int v57; // edi
  int v58; // edx
  unsigned int v59; // edx
  int v60; // r9d
  __int64 v61; // r10
  unsigned int v62; // ecx
  int v63; // r14d
  int v64; // edi
  _DWORD *v65; // rsi
  int v66; // r9d
  __int64 v67; // r10
  int v68; // edi
  unsigned int v69; // ecx
  int v70; // r14d
  __int64 v71; // rax
  unsigned int v72; // r12d
  unsigned __int64 v73; // rdx
  int v74; // r9d
  int *v75; // r8
  int v76; // eax
  int v77; // eax
  int v78; // r9d
  __int64 v79; // r8
  int *v80; // rdi
  unsigned int v81; // r10d
  int v82; // ecx
  int v83; // esi
  __int64 v84; // [rsp+8h] [rbp-A8h]
  char v86; // [rsp+1Ch] [rbp-94h]
  unsigned int v87; // [rsp+24h] [rbp-8Ch] BYREF
  unsigned int v88; // [rsp+28h] [rbp-88h] BYREF
  int v89; // [rsp+2Ch] [rbp-84h] BYREF
  _BYTE *v90; // [rsp+30h] [rbp-80h] BYREF
  __int64 v91; // [rsp+38h] [rbp-78h]
  _BYTE v92[16]; // [rsp+40h] [rbp-70h] BYREF
  _DWORD v93[24]; // [rsp+50h] [rbp-60h] BYREF

  v6 = a2;
  v90 = v92;
  v91 = 0x400000000LL;
  v87 = 0;
  v86 = 0;
  v84 = a1 + 112;
  while ( 1 )
  {
    v8 = *(_QWORD *)(a1 + 72);
    v9 = *(__int64 **)(a1 + 32);
    if ( v6 < 0 )
    {
      v10 = *(_QWORD *)(v9[7] + 16LL * (v6 & 0x7FFFFFFF) + 8);
    }
    else
    {
      v9 = (__int64 *)v9[38];
      v10 = v9[v6];
    }
    while ( 1 )
    {
      if ( !v10 )
        goto LABEL_8;
      if ( (*(_BYTE *)(v10 + 3) & 0x10) == 0 )
      {
        v11 = *(_BYTE *)(v10 + 4);
        if ( (v11 & 8) == 0 )
          break;
      }
      v10 = *(_QWORD *)(v10 + 32);
    }
    v32 = 0;
LABEL_29:
    if ( (v11 & 1) == 0 )
    {
      v33 = *(_QWORD *)(v10 + 16);
      if ( v8 != *(_QWORD *)(v33 + 24) )
        goto LABEL_8;
      if ( sub_2FF9BD0((_QWORD *)a1, v33, v6) )
        v32 = v10;
    }
    while ( 1 )
    {
      v10 = *(_QWORD *)(v10 + 32);
      if ( !v10 )
        break;
      while ( (*(_BYTE *)(v10 + 3) & 0x10) == 0 )
      {
        v11 = *(_BYTE *)(v10 + 4);
        if ( (v11 & 8) == 0 )
          goto LABEL_29;
        v10 = *(_QWORD *)(v10 + 32);
        if ( !v10 )
          goto LABEL_37;
      }
    }
LABEL_37:
    if ( !v32 )
      goto LABEL_8;
    v87 = 0;
    v34 = *(_QWORD *)(v32 + 16);
    v35 = *(_WORD *)(v34 + 68);
    if ( v35 == 20 || v35 == 12 || v35 == 9 )
    {
      v87 = *(_DWORD *)(*(_QWORD *)(v34 + 32) + 8LL);
      v39 = v87 - 1 <= 0x3FFFFFFE;
    }
    else
    {
      if ( !(unsigned __int8)sub_2FF8950(*(_QWORD *)(v32 + 16), v6, &v87) )
      {
        if ( (*(_BYTE *)(*(_QWORD *)(v34 + 16) + 27LL) & 2) == 0 )
          goto LABEL_8;
        v88 = -1;
        v36 = sub_2EAB0A0(v32);
        v37 = *(_QWORD *)(a1 + 8);
        v93[0] = v36;
        if ( !(*(unsigned __int8 (__fastcall **)(__int64, __int64, unsigned int *, _DWORD *))(*(_QWORD *)v37 + 264LL))(
                v37,
                v34,
                &v88,
                v93) )
          goto LABEL_8;
        v38 = *(_QWORD *)(v34 + 32) + 40LL * v88;
        if ( *(_BYTE *)v38
          || (*(_BYTE *)(v38 + 3) & 0x10) != 0
          || !(unsigned __int8)sub_2FF8950(v34, *(_DWORD *)(v38 + 8), &v87) )
        {
          goto LABEL_8;
        }
      }
      v39 = v87 - 1 <= 0x3FFFFFFE;
      if ( !v86 )
        goto LABEL_58;
    }
    v86 = *(_BYTE *)(a1 + 140);
    if ( !v86 )
      goto LABEL_57;
    v40 = *(__int64 **)(a1 + 120);
    a4 = *(unsigned int *)(a1 + 132);
    v9 = &v40[a4];
    if ( v40 != v9 )
    {
      while ( v34 != *v40 )
      {
        if ( v9 == ++v40 )
          goto LABEL_68;
      }
LABEL_8:
      v12 = v91;
      goto LABEL_9;
    }
LABEL_68:
    if ( (unsigned int)a4 < *(_DWORD *)(a1 + 128) )
    {
      *(_DWORD *)(a1 + 132) = a4 + 1;
      *v9 = v34;
      ++*(_QWORD *)(a1 + 112);
    }
    else
    {
LABEL_57:
      sub_C8CC70(v84, v34, (__int64)v9, a4, a5, a6);
      v86 = v41;
      if ( !v41 )
        goto LABEL_8;
    }
LABEL_58:
    v42 = *(unsigned int *)(a1 + 104);
    v43 = *(_QWORD *)(a1 + 88);
    if ( (_DWORD)v42 )
    {
      v44 = (v42 - 1) & (((unsigned int)v34 >> 9) ^ ((unsigned int)v34 >> 4));
      v45 = (__int64 *)(v43 + 16LL * v44);
      a5 = *v45;
      if ( v34 == *v45 )
      {
LABEL_60:
        if ( v45 != (__int64 *)(v43 + 16 * v42) )
          goto LABEL_8;
      }
      else
      {
        v56 = 1;
        while ( a5 != -4096 )
        {
          a6 = (unsigned int)(v56 + 1);
          v44 = (v42 - 1) & (v56 + v44);
          v45 = (__int64 *)(v43 + 16LL * v44);
          a5 = *v45;
          if ( v34 == *v45 )
            goto LABEL_60;
          v56 = a6;
        }
      }
    }
    if ( v39 )
      break;
    v46 = *(_DWORD *)(a1 + 232);
    a6 = a1 + 208;
    if ( v46 )
    {
      v47 = *(_QWORD *)(a1 + 216);
      v48 = 1;
      v49 = (v46 - 1) & (37 * v87);
      v50 = (unsigned int *)(v47 + 8LL * v49);
      v51 = 0;
      a5 = *v50;
      if ( v87 == (_DWORD)a5 )
      {
LABEL_64:
        v52 = (int *)(v50 + 1);
        goto LABEL_65;
      }
      while ( (_DWORD)a5 != -1 )
      {
        if ( (_DWORD)a5 == -2 && !v51 )
          v51 = v50;
        v49 = (v46 - 1) & (v48 + v49);
        v50 = (unsigned int *)(v47 + 8LL * v49);
        a5 = *v50;
        if ( v87 == (_DWORD)a5 )
          goto LABEL_64;
        ++v48;
      }
      if ( !v51 )
        v51 = v50;
      v57 = *(_DWORD *)(a1 + 224);
      ++*(_QWORD *)(a1 + 208);
      v58 = v57 + 1;
      if ( 4 * (v57 + 1) < 3 * v46 )
      {
        if ( v46 - *(_DWORD *)(a1 + 228) - v58 > v46 >> 3 )
          goto LABEL_84;
        sub_2FFACA0(a1 + 208, v46);
        v66 = *(_DWORD *)(a1 + 232);
        if ( !v66 )
        {
LABEL_142:
          ++*(_DWORD *)(a1 + 224);
          BUG();
        }
        a5 = v87;
        a6 = (unsigned int)(v66 - 1);
        v65 = 0;
        v67 = *(_QWORD *)(a1 + 216);
        v58 = *(_DWORD *)(a1 + 224) + 1;
        v68 = 1;
        v69 = a6 & (37 * v87);
        v51 = (_DWORD *)(v67 + 8LL * v69);
        v70 = *v51;
        if ( *v51 == v87 )
          goto LABEL_84;
        while ( v70 != -1 )
        {
          if ( v70 == -2 && !v65 )
            v65 = v51;
          v69 = a6 & (v68 + v69);
          v51 = (_DWORD *)(v67 + 8LL * v69);
          v70 = *v51;
          if ( v87 == *v51 )
            goto LABEL_84;
          ++v68;
        }
        goto LABEL_100;
      }
    }
    else
    {
      ++*(_QWORD *)(a1 + 208);
    }
    sub_2FFACA0(a1 + 208, 2 * v46);
    v60 = *(_DWORD *)(a1 + 232);
    if ( !v60 )
      goto LABEL_142;
    a5 = v87;
    a6 = (unsigned int)(v60 - 1);
    v61 = *(_QWORD *)(a1 + 216);
    v58 = *(_DWORD *)(a1 + 224) + 1;
    v62 = a6 & (37 * v87);
    v51 = (_DWORD *)(v61 + 8LL * v62);
    v63 = *v51;
    if ( *v51 == v87 )
      goto LABEL_84;
    v64 = 1;
    v65 = 0;
    while ( v63 != -1 )
    {
      if ( !v65 && v63 == -2 )
        v65 = v51;
      v62 = a6 & (v64 + v62);
      v51 = (_DWORD *)(v61 + 8LL * v62);
      v63 = *v51;
      if ( v87 == *v51 )
        goto LABEL_84;
      ++v64;
    }
LABEL_100:
    if ( v65 )
      v51 = v65;
LABEL_84:
    *(_DWORD *)(a1 + 224) = v58;
    if ( *v51 != -1 )
      --*(_DWORD *)(a1 + 228);
    v59 = v87;
    v51[1] = 0;
    v52 = v51 + 1;
    *(v52 - 1) = v59;
LABEL_65:
    *v52 = v6;
    v53 = (unsigned int)v91;
    a4 = HIDWORD(v91);
    v54 = v87;
    v55 = (unsigned int)v91 + 1LL;
    if ( v55 > HIDWORD(v91) )
    {
      sub_C8D5F0((__int64)&v90, v92, v55, 4u, a5, a6);
      v53 = (unsigned int)v91;
    }
    *(_DWORD *)&v90[4 * v53] = v54;
    v6 = v87;
    LODWORD(v91) = v91 + 1;
  }
  v71 = (unsigned int)v91;
  v72 = v87;
  v73 = (unsigned int)v91 + 1LL;
  if ( v73 > HIDWORD(v91) )
  {
    sub_C8D5F0((__int64)&v90, v92, v73, 4u, a5, a6);
    v71 = (unsigned int)v91;
  }
  *(_DWORD *)&v90[4 * v71] = v72;
  v12 = v91 + 1;
  LODWORD(v91) = v91 + 1;
LABEL_9:
  if ( v12 )
  {
    v13 = (unsigned __int64)v90;
    v14 = v12;
    v15 = v12 - 1;
    v16 = *(_DWORD *)&v90[4 * v14 - 4];
    LODWORD(v91) = v15;
    if ( !v15 )
    {
      v17 = v16;
LABEL_12:
      v89 = v17;
      v88 = a2;
      sub_2FFB080((__int64)v93, a1 + 240, (int *)&v88, &v89);
      goto LABEL_13;
    }
    while ( 2 )
    {
      v22 = *(_DWORD *)(a1 + 264);
      v23 = v15--;
      v17 = *(_DWORD *)(v13 + 4 * v23 - 4);
      LODWORD(v91) = v15;
      if ( !v22 )
      {
        ++*(_QWORD *)(a1 + 240);
        goto LABEL_21;
      }
      v18 = *(_QWORD *)(a1 + 248);
      v19 = (v22 - 1) & (37 * v17);
      v20 = (int *)(v18 + 8LL * v19);
      v21 = *v20;
      if ( v17 != *v20 )
      {
        v74 = 1;
        v75 = 0;
        while ( v21 != -1 )
        {
          if ( !v75 && v21 == -2 )
            v75 = v20;
          v19 = (v22 - 1) & (v74 + v19);
          v20 = (int *)(v18 + 8LL * v19);
          v21 = *v20;
          if ( v17 == *v20 )
            goto LABEL_17;
          ++v74;
        }
        v76 = *(_DWORD *)(a1 + 256);
        if ( v75 )
          v20 = v75;
        ++*(_QWORD *)(a1 + 240);
        v29 = v76 + 1;
        if ( 4 * v29 >= 3 * v22 )
        {
LABEL_21:
          sub_2FFACA0(a1 + 240, 2 * v22);
          v24 = *(_DWORD *)(a1 + 264);
          if ( !v24 )
            goto LABEL_141;
          v25 = v24 - 1;
          v26 = *(_QWORD *)(a1 + 248);
          v27 = v25 & (37 * v17);
          v20 = (int *)(v26 + 8LL * v27);
          v28 = *v20;
          v29 = *(_DWORD *)(a1 + 256) + 1;
          if ( v17 != *v20 )
          {
            v30 = 1;
            v31 = 0;
            while ( v28 != -1 )
            {
              if ( !v31 && v28 == -2 )
                v31 = v20;
              v27 = v25 & (v30 + v27);
              v20 = (int *)(v26 + 8LL * v27);
              v28 = *v20;
              if ( v17 == *v20 )
                goto LABEL_112;
              ++v30;
            }
            if ( v31 )
              v20 = v31;
          }
        }
        else if ( v22 - *(_DWORD *)(a1 + 260) - v29 <= v22 >> 3 )
        {
          sub_2FFACA0(a1 + 240, v22);
          v77 = *(_DWORD *)(a1 + 264);
          if ( !v77 )
          {
LABEL_141:
            ++*(_DWORD *)(a1 + 256);
            BUG();
          }
          v78 = v77 - 1;
          v79 = *(_QWORD *)(a1 + 248);
          v80 = 0;
          v81 = (v77 - 1) & (37 * v17);
          v82 = 1;
          v20 = (int *)(v79 + 8LL * v81);
          v29 = *(_DWORD *)(a1 + 256) + 1;
          v83 = *v20;
          if ( v17 != *v20 )
          {
            while ( v83 != -1 )
            {
              if ( !v80 && v83 == -2 )
                v80 = v20;
              v81 = v78 & (v82 + v81);
              v20 = (int *)(v79 + 8LL * v81);
              v83 = *v20;
              if ( v17 == *v20 )
                goto LABEL_112;
              ++v82;
            }
            if ( v80 )
              v20 = v80;
          }
        }
LABEL_112:
        *(_DWORD *)(a1 + 256) = v29;
        if ( *v20 != -1 )
          --*(_DWORD *)(a1 + 260);
        *v20 = v17;
        v20[1] = v16;
        v15 = v91;
      }
LABEL_17:
      if ( !v15 )
        goto LABEL_12;
      v13 = (unsigned __int64)v90;
      v16 = v17;
      continue;
    }
  }
LABEL_13:
  if ( v90 != v92 )
    _libc_free((unsigned __int64)v90);
}
