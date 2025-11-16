// Function: sub_3910BC0
// Address: 0x3910bc0
//
void __fastcall sub_3910BC0(__int64 a1, __int64 **a2, __int64 a3)
{
  __int64 v3; // r13
  __int64 v4; // rbx
  __int64 v5; // rax
  unsigned __int64 v6; // rdx
  unsigned __int64 v7; // r12
  _DWORD *v8; // rax
  unsigned __int64 v9; // r8
  __int64 v10; // rax
  __int64 v11; // rdx
  __int64 v12; // r15
  __int64 v13; // r9
  int v14; // r8d
  int v15; // r11d
  char v16; // si
  int v17; // r13d
  int v18; // r14d
  __int64 v19; // rax
  __int64 v20; // rdx
  __int64 v21; // rax
  __int64 v22; // rcx
  unsigned int v23; // ecx
  int v24; // r8d
  unsigned int v25; // eax
  __int64 v26; // rcx
  unsigned int v27; // r8d
  unsigned int v28; // r9d
  __int64 v29; // rax
  __int64 v30; // rdx
  int v31; // ecx
  __int64 v32; // rax
  __int64 v33; // rdi
  unsigned int v34; // r14d
  int *v35; // rdx
  int v36; // r10d
  __int64 v37; // rdx
  unsigned int v38; // esi
  __int64 v39; // rax
  int v40; // r8d
  int v41; // r9d
  __int64 v42; // rdx
  unsigned int v43; // r14d
  __int64 v44; // rax
  __int64 v45; // rcx
  int v46; // r8d
  __int64 v47; // r9
  __int64 v48; // r12
  __int64 v49; // rdx
  __int64 v50; // rax
  __int64 v51; // rdx
  _DWORD *v52; // rdi
  unsigned int *v53; // rax
  unsigned int *v54; // r15
  unsigned int v55; // esi
  unsigned int *v56; // r14
  __int64 v57; // rbx
  unsigned __int64 v58; // r13
  unsigned __int64 v59; // rax
  unsigned __int64 v60; // rdx
  int v61; // edx
  unsigned int v62; // eax
  __int64 v63; // rcx
  int v64; // r9d
  int v65; // r8d
  int v66; // r11d
  unsigned int v67; // r13d
  __int64 v68; // rax
  __int64 v69; // rdx
  __int64 v70; // r13
  unsigned __int64 v71; // rax
  __int64 v72; // r13
  unsigned __int64 v73; // rax
  __int64 v74; // rax
  unsigned int v75; // eax
  int v76; // [rsp+Ch] [rbp-74h]
  __int64 v77; // [rsp+18h] [rbp-68h]
  __int64 v78; // [rsp+20h] [rbp-60h]
  _DWORD *v79; // [rsp+28h] [rbp-58h]
  __int64 v80; // [rsp+30h] [rbp-50h]
  int v82; // [rsp+40h] [rbp-40h]
  unsigned int v83; // [rsp+40h] [rbp-40h]
  int v84; // [rsp+40h] [rbp-40h]
  int v85; // [rsp+40h] [rbp-40h]
  __int64 v86; // [rsp+40h] [rbp-40h]
  unsigned __int64 v87; // [rsp+48h] [rbp-38h]
  __int64 v88; // [rsp+48h] [rbp-38h]
  unsigned int v89; // [rsp+48h] [rbp-38h]
  unsigned int v90; // [rsp+48h] [rbp-38h]
  unsigned int v91; // [rsp+48h] [rbp-38h]
  __int64 v92; // [rsp+48h] [rbp-38h]
  __int64 v93; // [rsp+48h] [rbp-38h]
  int v94; // [rsp+48h] [rbp-38h]
  int v95; // [rsp+48h] [rbp-38h]
  int v96; // [rsp+48h] [rbp-38h]
  _QWORD *v97; // [rsp+48h] [rbp-38h]
  int v98; // [rsp+48h] [rbp-38h]
  int v99; // [rsp+48h] [rbp-38h]
  unsigned int v100; // [rsp+48h] [rbp-38h]

  v3 = a1;
  v4 = a3;
  v5 = sub_3910890(a1, *(_DWORD *)(a3 + 48));
  v7 = v6;
  v87 = v5;
  v8 = sub_390FE40(a1, *(_DWORD *)(v4 + 48));
  v9 = v87;
  v79 = v8;
  if ( v8[10] )
  {
    v52 = v8;
    v53 = (unsigned int *)*((_QWORD *)v8 + 4);
    v54 = &v53[4 * v52[12]];
    if ( v53 != v54 )
    {
      while ( 1 )
      {
        v55 = *v53;
        v56 = v53;
        if ( *v53 <= 0xFFFFFFFD )
          break;
        v53 += 4;
        if ( v54 == v53 )
          goto LABEL_2;
      }
      if ( v54 != v53 )
      {
        v93 = v4;
        v57 = v3;
        v58 = v9;
        while ( 1 )
        {
          v59 = sub_3910890(v57, v55);
          if ( v58 > v59 )
            v58 = v59;
          if ( v7 < v60 )
            v7 = v60;
          v56 += 4;
          if ( v56 == v54 )
            break;
          while ( *v56 > 0xFFFFFFFD )
          {
            v56 += 4;
            if ( v54 == v56 )
              goto LABEL_61;
          }
          if ( v54 == v56 )
            break;
          v55 = *v56;
        }
LABEL_61:
        v9 = v58;
        v3 = v57;
        v4 = v93;
      }
    }
  }
LABEL_2:
  if ( v7 <= v9 )
    return;
  v10 = sub_3910900(v3, v9, v7);
  if ( !v11 )
    return;
  v12 = v10;
  v13 = *(_QWORD *)(v4 + 64);
  v14 = *(_DWORD *)(v4 + 52);
  v15 = *(_DWORD *)(v4 + 56);
  v77 = v4 + 80;
  *(_DWORD *)(v4 + 88) = 0;
  v80 = v10 + 24 * v11;
  if ( v10 == v80 )
    goto LABEL_44;
  v16 = 0;
  v78 = v3;
  do
  {
    v31 = *(_DWORD *)v12;
    if ( *(_DWORD *)(v4 + 48) == *(_DWORD *)v12 )
    {
      v17 = *(_DWORD *)(v12 + 4);
      v18 = *(_DWORD *)(v12 + 8);
      if ( !v16 )
        goto LABEL_26;
      goto LABEL_8;
    }
    v32 = (unsigned int)v79[12];
    if ( (_DWORD)v32 )
    {
      v33 = *((_QWORD *)v79 + 4);
      v34 = (v32 - 1) & (37 * v31);
      v35 = (int *)(v33 + 16LL * v34);
      v36 = *v35;
      if ( v31 == *v35 )
      {
LABEL_24:
        if ( v35 != (int *)(v33 + 16 * v32) )
        {
          v17 = v35[1];
          v18 = v35[2];
          if ( !v16 )
          {
LABEL_26:
            if ( v17 == v14 )
            {
              v23 = v18 - v15;
              v24 = 2 * (v18 - v15);
              if ( v18 - v15 < 0 )
                goto LABEL_28;
              goto LABEL_13;
            }
            goto LABEL_9;
          }
LABEL_8:
          if ( v17 == v14 )
          {
            if ( v18 == v15 )
              goto LABEL_19;
LABEL_12:
            v23 = v18 - v15;
            v24 = 2 * (v18 - v15);
            if ( v18 - v15 < 0 )
LABEL_28:
              v24 = 2 * (v15 - v18) + 1;
LABEL_13:
            v83 = v24;
            v89 = v23;
            v25 = sub_390F830(a2, v13, *(_QWORD *)(v12 + 16));
            v26 = v89;
            v27 = v83;
            v28 = v25;
            if ( !v25 && v89 )
            {
              v29 = *(unsigned int *)(v4 + 88);
              if ( (unsigned int)v29 >= *(_DWORD *)(v4 + 92) )
              {
                sub_16CD150(v4 + 80, (const void *)(v4 + 96), 0, 1, v83, 0);
                v29 = *(unsigned int *)(v4 + 88);
                v27 = v83;
              }
              v30 = *(_QWORD *)(v4 + 80);
              *(_BYTE *)(v30 + v29) = 6;
              ++*(_DWORD *)(v4 + 88);
              sub_390FA80(v27, v4 + 80, v30, v26, v27, v28);
              goto LABEL_18;
            }
            v37 = *(unsigned int *)(v4 + 88);
            v38 = *(_DWORD *)(v4 + 92);
            v39 = v37;
            if ( v83 <= 7 && v28 <= 0xF )
            {
              v40 = 16 * v83;
              v41 = (16 * v83) | v28;
              if ( v38 <= (unsigned int)v37 )
              {
                v98 = v41;
                sub_16CD150(v4 + 80, (const void *)(v4 + 96), 0, 1, v40, v41);
                v37 = *(unsigned int *)(v4 + 88);
                v41 = v98;
              }
              *(_BYTE *)(*(_QWORD *)(v4 + 80) + v37) = 11;
              ++*(_DWORD *)(v4 + 88);
              sub_390FA80(v41, v4 + 80, v37, v26, v40, v41);
              goto LABEL_18;
            }
            if ( v89 )
            {
              if ( v38 <= (unsigned int)v37 )
              {
                v100 = v28;
                sub_16CD150(v4 + 80, (const void *)(v4 + 96), 0, 1, v83, v28);
                v37 = *(unsigned int *)(v4 + 88);
                v27 = v83;
                v28 = v100;
              }
              v90 = v28;
              *(_BYTE *)(*(_QWORD *)(v4 + 80) + v37) = 6;
              ++*(_DWORD *)(v4 + 88);
              sub_390FA80(v27, v4 + 80, v37, v26, v27, v28);
              v39 = *(unsigned int *)(v4 + 88);
              v28 = v90;
              if ( (unsigned int)v39 < *(_DWORD *)(v4 + 92) )
                goto LABEL_36;
            }
            else if ( (unsigned int)v37 < *(_DWORD *)(v4 + 92) )
            {
LABEL_36:
              v42 = *(_QWORD *)(v4 + 80);
              *(_BYTE *)(v42 + v39) = 3;
              ++*(_DWORD *)(v4 + 88);
              sub_390FA80(v28, v4 + 80, v42, v26, v27, v28);
LABEL_18:
              v13 = *(_QWORD *)(v12 + 16);
              v15 = v18;
              v14 = v17;
              v16 = 1;
              goto LABEL_19;
            }
            v91 = v28;
            sub_16CD150(v4 + 80, (const void *)(v4 + 96), 0, 1, v27, v28);
            v39 = *(unsigned int *)(v4 + 88);
            v28 = v91;
            goto LABEL_36;
          }
LABEL_9:
          v19 = *(_QWORD *)(*(_QWORD *)(v78 + 72) + 32LL * (unsigned int)(v17 - 1) + 24);
          *(_BYTE *)(v19 + 8) |= 4u;
          v20 = *(unsigned int *)(v4 + 88);
          v21 = *(_QWORD *)(*(_QWORD *)(v19 + 24) + 16LL);
          if ( (unsigned int)v20 >= *(_DWORD *)(v4 + 92) )
          {
            v76 = v15;
            v86 = v13;
            v96 = v21;
            sub_16CD150(v4 + 80, (const void *)(v4 + 96), 0, 1, v14, v13);
            v20 = *(unsigned int *)(v4 + 88);
            v15 = v76;
            v13 = v86;
            LODWORD(v21) = v96;
          }
          v22 = *(_QWORD *)(v4 + 80);
          v82 = v15;
          v88 = v13;
          *(_BYTE *)(v22 + v20) = 5;
          ++*(_DWORD *)(v4 + 88);
          sub_390FA80(v21, v4 + 80, v20, v22, v14, v13);
          v15 = v82;
          v13 = v88;
          goto LABEL_12;
        }
      }
      else
      {
        v61 = 1;
        while ( v36 != -1 )
        {
          v34 = (v32 - 1) & (v61 + v34);
          v99 = v61 + 1;
          v35 = (int *)(v33 + 16LL * v34);
          v36 = *v35;
          if ( v31 == *v35 )
            goto LABEL_24;
          v61 = v99;
        }
      }
    }
    if ( v16 )
    {
      v84 = v15;
      v94 = v14;
      v62 = sub_390F830(a2, v13, *(_QWORD *)(v12 + 16));
      v65 = v94;
      v66 = v84;
      v67 = v62;
      v68 = *(unsigned int *)(v4 + 88);
      if ( (unsigned int)v68 >= *(_DWORD *)(v4 + 92) )
      {
        sub_16CD150(v4 + 80, (const void *)(v4 + 96), 0, 1, v94, v64);
        v68 = *(unsigned int *)(v4 + 88);
        v66 = v84;
        v65 = v94;
      }
      v69 = *(_QWORD *)(v4 + 80);
      v85 = v66;
      v95 = v65;
      *(_BYTE *)(v69 + v68) = 4;
      ++*(_DWORD *)(v4 + 88);
      sub_390FA80(v67, v4 + 80, v69, v63, v65, v64);
      v13 = *(_QWORD *)(v12 + 16);
      v15 = v85;
      v16 = 0;
      v14 = v95;
    }
    else
    {
      v16 = 0;
    }
LABEL_19:
    v12 += 24;
  }
  while ( v12 != v80 && *(_DWORD *)(v4 + 88) <= 0xFEEBu );
  v3 = v78;
LABEL_44:
  v92 = v13;
  v43 = sub_390F830(a2, v13, *(_QWORD *)(v4 + 72));
  v44 = sub_3910900(v3, v7, v7 + 1);
  v47 = v92;
  v48 = v44;
  if ( v49 )
  {
    v70 = *(_QWORD *)(v44 + 16);
    v71 = *(_QWORD *)v70 & 0xFFFFFFFFFFFFFFF8LL;
    if ( v71
      || (*(_BYTE *)(v70 + 9) & 0xC) == 8
      && (*(_BYTE *)(v70 + 8) |= 4u,
          v71 = (unsigned __int64)sub_38CE440(*(_QWORD *)(v70 + 24)),
          v47 = v92,
          *(_QWORD *)v70 = v71 | *(_QWORD *)v70 & 7LL,
          v71) )
    {
      v72 = *(_QWORD *)(v71 + 24);
    }
    else
    {
      v72 = 0;
    }
    v73 = *(_QWORD *)v47 & 0xFFFFFFFFFFFFFFF8LL;
    if ( v73
      || (*(_BYTE *)(v47 + 9) & 0xC) == 8
      && (*(_BYTE *)(v47 + 8) |= 4u,
          v97 = (_QWORD *)v47,
          v73 = (unsigned __int64)sub_38CE440(*(_QWORD *)(v47 + 24)),
          v47 = (__int64)v97,
          *v97 = v73 | *v97 & 7LL,
          v73) )
    {
      v74 = *(_QWORD *)(v73 + 24);
    }
    else
    {
      v74 = 0;
    }
    if ( v74 == v72 )
    {
      v75 = sub_390F830(a2, v47, *(_QWORD *)(v48 + 16));
      if ( v43 > v75 )
        v43 = v75;
    }
  }
  v50 = *(unsigned int *)(v4 + 88);
  if ( (unsigned int)v50 >= *(_DWORD *)(v4 + 92) )
  {
    sub_16CD150(v77, (const void *)(v4 + 96), 0, 1, v46, v47);
    v50 = *(unsigned int *)(v4 + 88);
  }
  v51 = *(_QWORD *)(v4 + 80);
  *(_BYTE *)(v51 + v50) = 4;
  ++*(_DWORD *)(v4 + 88);
  sub_390FA80(v43, v77, v51, v45, v46, v47);
}
