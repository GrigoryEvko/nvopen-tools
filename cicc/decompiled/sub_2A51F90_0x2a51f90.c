// Function: sub_2A51F90
// Address: 0x2a51f90
//
__int64 __fastcall sub_2A51F90(
        _QWORD *a1,
        __int64 a2,
        __int64 a3,
        unsigned __int64 a4,
        __int64 a5,
        __int64 a6,
        __int64 a7,
        __int64 a8,
        char a9)
{
  __int64 v9; // r13
  __int64 v10; // r12
  __int64 v11; // rbx
  unsigned __int8 *v12; // rax
  __int64 v13; // rax
  __int64 v14; // r12
  __int64 v15; // r13
  __int64 v16; // rax
  __int64 v17; // r15
  char v18; // al
  __int64 v19; // rax
  __int64 v20; // rdx
  __int64 v21; // r14
  int v22; // r14d
  __int64 v23; // rax
  __int64 v24; // rdx
  __int64 v25; // rax
  __int64 v26; // rdx
  __int64 v27; // rsi
  __int64 v28; // rcx
  __int64 v29; // rax
  __int64 v30; // rdx
  __int64 v31; // rdx
  __int64 result; // rax
  __int64 v33; // rdx
  __int64 v34; // r8
  __int64 v35; // r9
  int v36; // ecx
  __int64 v37; // rsi
  int v38; // ecx
  unsigned int v39; // edx
  __int64 *v40; // rax
  __int64 v41; // r8
  __int64 v42; // rax
  __int64 v43; // rax
  __int64 v44; // r8
  __int64 v45; // r9
  __int64 v46; // rax
  __int64 v47; // r9
  __int64 v48; // rax
  __int64 v49; // rax
  __int64 *v50; // rax
  __int64 v51; // rcx
  __int64 *v52; // r15
  __int64 *v53; // rbx
  __int64 v54; // r13
  __int64 v55; // rax
  int v56; // eax
  __int64 *v57; // r15
  __int64 *v58; // r13
  __int64 v59; // r9
  __int64 v60; // rax
  _QWORD *v61; // rbx
  char v62; // al
  __int64 v63; // rdi
  int v64; // eax
  __int64 v65; // rsi
  __int64 v66; // rcx
  int v67; // edi
  unsigned int v68; // edx
  __int64 *v69; // rax
  __int64 v70; // r8
  __int64 v71; // rax
  bool v72; // al
  unsigned __int8 *v73; // rdi
  int v74; // eax
  int v75; // edi
  __int64 v76; // r14
  __int64 v77; // rax
  int v78; // eax
  int v79; // r9d
  const void *v80; // [rsp+0h] [rbp-240h]
  __int64 v84; // [rsp+20h] [rbp-220h]
  __int64 v86; // [rsp+30h] [rbp-210h]
  unsigned int v87; // [rsp+38h] [rbp-208h]
  char v88; // [rsp+3Fh] [rbp-201h]
  unsigned __int8 *v90; // [rsp+48h] [rbp-1F8h]
  __int64 v91; // [rsp+50h] [rbp-1F0h]
  __int64 v92; // [rsp+50h] [rbp-1F0h]
  __int64 v93; // [rsp+58h] [rbp-1E8h]
  __int64 v94[60]; // [rsp+60h] [rbp-1E0h] BYREF

  v9 = 0;
  v10 = a2;
  v11 = a3;
  v91 = *(_QWORD *)(a2 + 544);
  if ( !a9 )
  {
LABEL_2:
    v88 = 1;
    v12 = *(unsigned __int8 **)(v91 - 64);
    v90 = v12;
    if ( *v12 <= 0x1Cu )
      v88 = sub_98ED70(v12, 0, 0, 0, 0) ^ 1;
    goto LABEL_4;
  }
  v73 = *(unsigned __int8 **)(*(_QWORD *)(a2 + 544) - 64LL);
  v90 = v73;
  if ( *v73 != 61 )
  {
    v9 = sub_2A4E920(v91, a2, a3, a4, a5, a6);
    goto LABEL_2;
  }
  v88 = a9;
  v9 = *((_QWORD *)v73 - 4);
LABEL_4:
  v87 = -1;
  v13 = *(_QWORD *)(v91 + 40);
  *(_DWORD *)(a2 + 280) = 0;
  v84 = v13;
  v86 = v9 + 16;
  v80 = (const void *)(a2 + 288);
  if ( !a1[2] )
    goto LABEL_67;
  v14 = v9;
  v15 = a1[2];
  do
  {
    while ( 1 )
    {
      while ( 1 )
      {
        v16 = v15;
        v15 = *(_QWORD *)(v15 + 8);
        v17 = *(_QWORD *)(v16 + 24);
        if ( a9 )
          break;
LABEL_30:
        if ( v91 == v17 )
        {
LABEL_26:
          if ( !v15 )
            goto LABEL_27;
        }
        else
        {
          if ( !v88 )
            goto LABEL_34;
          v33 = *(_QWORD *)(v17 + 40);
          if ( v84 == v33 )
          {
            if ( v87 == -1 )
              v87 = sub_2A4E220(v11, v91);
            if ( (unsigned int)sub_2A4E220(v11, v17) >= v87 )
            {
LABEL_34:
              if ( a9 )
                goto LABEL_42;
              if ( (unsigned __int8 *)v17 == v90 )
                v90 = (unsigned __int8 *)sub_ACADE0(*(__int64 ***)(v17 + 8));
              sub_2A4C510(v17, v90, a4, a6, a5);
              sub_BD84D0(v17, (__int64)v90);
              sub_B43D60((_QWORD *)v17);
              v36 = *(_DWORD *)(v11 + 24);
              v37 = *(_QWORD *)(v11 + 8);
              if ( !v36 )
                goto LABEL_26;
              v38 = v36 - 1;
              v39 = v38 & (((unsigned int)v17 >> 9) ^ ((unsigned int)v17 >> 4));
              v40 = (__int64 *)(v37 + 16LL * v39);
              v41 = *v40;
              if ( v17 != *v40 )
              {
                v74 = 1;
                while ( v41 != -4096 )
                {
                  v75 = v74 + 1;
                  v39 = v38 & (v74 + v39);
                  v40 = (__int64 *)(v37 + 16LL * v39);
                  v41 = *v40;
                  if ( v17 == *v40 )
                    goto LABEL_39;
                  v74 = v75;
                }
                goto LABEL_26;
              }
LABEL_39:
              *v40 = -8192;
              --*(_DWORD *)(v11 + 16);
              ++*(_DWORD *)(v11 + 20);
              if ( !v15 )
                goto LABEL_27;
            }
            else
            {
              v46 = *(unsigned int *)(a2 + 280);
              if ( v46 + 1 > (unsigned __int64)*(unsigned int *)(a2 + 284) )
              {
                sub_C8D5F0(a2 + 272, v80, v46 + 1, 8u, v44, v45);
                v46 = *(unsigned int *)(a2 + 280);
              }
              *(_QWORD *)(*(_QWORD *)(a2 + 272) + 8 * v46) = v84;
              ++*(_DWORD *)(a2 + 280);
              if ( !v15 )
                goto LABEL_27;
            }
          }
          else
          {
            if ( (unsigned __int8)sub_B19720(a5, v84, v33) )
              goto LABEL_34;
            v76 = *(_QWORD *)(v17 + 40);
            v77 = *(unsigned int *)(a2 + 280);
            if ( v77 + 1 > (unsigned __int64)*(unsigned int *)(a2 + 284) )
            {
              sub_C8D5F0(a2 + 272, v80, v77 + 1, 8u, v34, v35);
              v77 = *(unsigned int *)(a2 + 280);
            }
            *(_QWORD *)(*(_QWORD *)(a2 + 272) + 8 * v77) = v76;
            ++*(_DWORD *)(a2 + 280);
            if ( !v15 )
              goto LABEL_27;
          }
        }
      }
      v18 = *(_BYTE *)v17;
      if ( *(_BYTE *)v17 == 85 )
      {
        if ( *(char *)(v17 + 7) < 0 )
        {
          v19 = sub_BD2BC0(v17);
          v21 = v19 + v20;
          if ( *(char *)(v17 + 7) >= 0 )
          {
            if ( (unsigned int)(v21 >> 4) )
LABEL_111:
              BUG();
          }
          else if ( (unsigned int)((v21 - sub_BD2BC0(v17)) >> 4) )
          {
            if ( *(char *)(v17 + 7) >= 0 )
              goto LABEL_111;
            v22 = *(_DWORD *)(sub_BD2BC0(v17) + 8);
            if ( *(char *)(v17 + 7) >= 0 )
              BUG();
            v23 = sub_BD2BC0(v17);
            v25 = 32LL * (unsigned int)(*(_DWORD *)(v23 + v24 - 4) - v22);
            goto LABEL_14;
          }
        }
        v25 = 0;
LABEL_14:
        v26 = *(_DWORD *)(v17 + 4) & 0x7FFFFFF;
        v27 = (32 * v26 - 32 - v25) >> 5;
        if ( (_DWORD)v27 )
        {
          v28 = 0;
          while ( 1 )
          {
            v29 = v17 + 32 * (v28 - v26);
            if ( !*(_QWORD *)v29 || a1 != *(_QWORD **)v29 )
              goto LABEL_16;
            v30 = *(_QWORD *)(v29 + 8);
            **(_QWORD **)(v29 + 16) = v30;
            if ( v30 )
              *(_QWORD *)(v30 + 16) = *(_QWORD *)(v29 + 16);
            *(_QWORD *)v29 = v14;
            if ( v14 )
            {
              v31 = *(_QWORD *)(v14 + 16);
              *(_QWORD *)(v29 + 8) = v31;
              if ( v31 )
                *(_QWORD *)(v31 + 16) = v29 + 8;
              ++v28;
              *(_QWORD *)(v29 + 16) = v86;
              *(_QWORD *)(v14 + 16) = v29;
              if ( (unsigned int)v27 == v28 )
                goto LABEL_26;
            }
            else
            {
LABEL_16:
              if ( (unsigned int)v27 == ++v28 )
                goto LABEL_26;
            }
            v26 = *(_DWORD *)(v17 + 4) & 0x7FFFFFF;
          }
        }
        goto LABEL_26;
      }
      if ( v18 != 78 )
        break;
LABEL_42:
      if ( *(_QWORD *)(v17 - 32) )
      {
        v42 = *(_QWORD *)(v17 - 24);
        **(_QWORD **)(v17 - 16) = v42;
        if ( v42 )
          *(_QWORD *)(v42 + 16) = *(_QWORD *)(v17 - 16);
      }
      *(_QWORD *)(v17 - 32) = v14;
      if ( !v14 )
        goto LABEL_26;
      v43 = *(_QWORD *)(v14 + 16);
      *(_QWORD *)(v17 - 24) = v43;
      if ( v43 )
        *(_QWORD *)(v43 + 16) = v17 - 24;
      *(_QWORD *)(v17 - 16) = v86;
      *(_QWORD *)(v14 + 16) = v17 - 32;
      if ( !v15 )
        goto LABEL_27;
    }
    if ( v18 != 63 )
      goto LABEL_30;
    v47 = v17 - 32LL * (*(_DWORD *)(v17 + 4) & 0x7FFFFFF);
    if ( *(_QWORD *)v47 )
    {
      v48 = *(_QWORD *)(v47 + 8);
      **(_QWORD **)(v47 + 16) = v48;
      if ( v48 )
        *(_QWORD *)(v48 + 16) = *(_QWORD *)(v47 + 16);
    }
    *(_QWORD *)v47 = v14;
    if ( !v14 )
      goto LABEL_26;
    v49 = *(_QWORD *)(v14 + 16);
    *(_QWORD *)(v47 + 8) = v49;
    if ( v49 )
      *(_QWORD *)(v49 + 16) = v47 + 8;
    *(_QWORD *)(v47 + 16) = v86;
    *(_QWORD *)(v14 + 16) = v47;
  }
  while ( v15 );
LABEL_27:
  v10 = a2;
  result = 0;
  if ( *(_DWORD *)(a2 + 280) )
    return result;
LABEL_67:
  v50 = (__int64 *)sub_B43CA0((__int64)a1);
  sub_AE0470((__int64)v94, v50, 0, 0);
  sub_2A518A0(v10 + 616, *(_QWORD *)(v10 + 544), v94, a7, a8);
  v51 = *(_QWORD *)(v10 + 568);
  if ( v51 == v51 + 8LL * *(unsigned int *)(v10 + 576) )
    goto LABEL_80;
  v93 = v11;
  v52 = (__int64 *)(v51 + 8LL * *(unsigned int *)(v10 + 576));
  v53 = *(__int64 **)(v10 + 568);
  while ( 2 )
  {
    while ( 2 )
    {
      v54 = *v53;
      v55 = *(_QWORD *)(*v53 - 32);
      if ( !v55 || *(_BYTE *)v55 || *(_QWORD *)(v55 + 24) != *(_QWORD *)(v54 + 80) )
        BUG();
      v56 = *(_DWORD *)(v55 + 36);
      if ( v56 == 69 )
      {
        sub_F519F0(*v53, *(_QWORD *)(v10 + 544), v94);
        sub_B43D60((_QWORD *)v54);
        goto LABEL_71;
      }
      if ( v56 != 71
        || !sub_AF4730(*(_QWORD *)(*(_QWORD *)(v54 + 32 * (2LL - (*(_DWORD *)(v54 + 4) & 0x7FFFFFF))) + 24LL)) )
      {
        if ( sub_AF4730(*(_QWORD *)(*(_QWORD *)(v54 + 32 * (2LL - (*(_DWORD *)(v54 + 4) & 0x7FFFFFF))) + 24LL)) )
          sub_B43D60((_QWORD *)v54);
LABEL_71:
        if ( v52 == ++v53 )
          goto LABEL_79;
        continue;
      }
      break;
    }
    ++v53;
    sub_F51AF0(v54, *(_QWORD *)(v10 + 544), v94);
    sub_B43D60((_QWORD *)v54);
    if ( v52 != v53 )
      continue;
    break;
  }
LABEL_79:
  v11 = v93;
LABEL_80:
  v57 = *(__int64 **)(v10 + 592);
  if ( v57 != &v57[*(unsigned int *)(v10 + 600)] )
  {
    v92 = v11;
    v58 = &v57[*(unsigned int *)(v10 + 600)];
    do
    {
      while ( 1 )
      {
        v61 = (_QWORD *)*v57;
        v62 = *(_BYTE *)(*v57 + 64);
        if ( v62 )
          break;
        v63 = *v57++;
        sub_F51C80(v63, *(_QWORD *)(v10 + 544), v94);
        sub_B14290(v61);
        if ( v58 == v57 )
          goto LABEL_88;
      }
      v59 = (__int64)(v61 + 10);
      if ( v62 == 1 && (v71 = sub_B11F60((__int64)(v61 + 10)), v72 = sub_AF4730(v71), v59 = (__int64)(v61 + 10), v72) )
      {
        sub_F51DC0((__int64)v61, *(_QWORD *)(v10 + 544), v94);
        sub_B14290(v61);
      }
      else
      {
        v60 = sub_B11F60(v59);
        if ( sub_AF4730(v60) )
          sub_B14290(v61);
      }
      ++v57;
    }
    while ( v58 != v57 );
LABEL_88:
    v11 = v92;
  }
  sub_AE94E0((__int64)a1);
  sub_B43D60(*(_QWORD **)(v10 + 544));
  v64 = *(_DWORD *)(v11 + 24);
  v65 = *(_QWORD *)(v11 + 8);
  v66 = *(_QWORD *)(v10 + 544);
  if ( v64 )
  {
    v67 = v64 - 1;
    v68 = (v64 - 1) & (((unsigned int)v66 >> 9) ^ ((unsigned int)v66 >> 4));
    v69 = (__int64 *)(v65 + 16LL * v68);
    v70 = *v69;
    if ( v66 == *v69 )
    {
LABEL_91:
      *v69 = -8192;
      --*(_DWORD *)(v11 + 16);
      ++*(_DWORD *)(v11 + 20);
    }
    else
    {
      v78 = 1;
      while ( v70 != -4096 )
      {
        v79 = v78 + 1;
        v68 = v67 & (v78 + v68);
        v69 = (__int64 *)(v65 + 16LL * v68);
        v70 = *v69;
        if ( v66 == *v69 )
          goto LABEL_91;
        v78 = v79;
      }
    }
  }
  sub_B43D60(a1);
  sub_AE9130((__int64)v94, v65);
  return 1;
}
