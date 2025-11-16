// Function: sub_13EDBA0
// Address: 0x13edba0
//
int *__fastcall sub_13EDBA0(int *a1, __int64 a2, __int64 a3, char a4, __int64 a5)
{
  unsigned int v8; // r13d
  __int64 v9; // rdx
  unsigned int v10; // eax
  unsigned int *v11; // rsi
  __int64 v12; // r10
  int v14; // esi
  unsigned __int8 v15; // al
  __int64 v16; // r14
  unsigned int v17; // ecx
  __int64 *v18; // rdi
  __int64 v19; // rax
  _BYTE *v20; // r11
  __int64 v21; // r14
  unsigned __int8 v22; // al
  int v23; // r10d
  char v24; // al
  int v25; // eax
  _BYTE *v26; // rax
  char v27; // al
  int v28; // eax
  int v29; // ecx
  __int64 v30; // rsi
  unsigned int v31; // edx
  int v32; // eax
  __int64 v33; // r10
  int v34; // r11d
  int v35; // r11d
  __int64 *v36; // r10
  int v37; // eax
  int v38; // eax
  int v39; // edx
  __int64 v40; // rsi
  int v41; // r11d
  __int64 *v42; // r10
  unsigned int v43; // r12d
  __int64 v44; // rcx
  unsigned int v45; // r10d
  char v46; // cl
  __int64 v47; // r8
  unsigned __int8 v48; // al
  unsigned int v49; // eax
  __int64 v50; // r8
  unsigned int v51; // eax
  __int64 v52; // rax
  unsigned int v53; // edx
  __int64 v54; // rax
  unsigned int v55; // eax
  unsigned int v56; // eax
  __int64 v57; // rax
  __int64 v58; // rax
  __int64 v59; // rdx
  __int64 v60; // rax
  int v61; // r12d
  __int64 *v62; // r11
  __int64 v63; // [rsp+8h] [rbp-108h]
  char v64; // [rsp+10h] [rbp-100h]
  __int64 v65; // [rsp+10h] [rbp-100h]
  __int64 v66; // [rsp+10h] [rbp-100h]
  __int64 v67; // [rsp+10h] [rbp-100h]
  __int64 v68; // [rsp+10h] [rbp-100h]
  __int64 v69; // [rsp+18h] [rbp-F8h]
  int v70; // [rsp+18h] [rbp-F8h]
  char v71; // [rsp+18h] [rbp-F8h]
  char v72; // [rsp+18h] [rbp-F8h]
  char v73; // [rsp+18h] [rbp-F8h]
  char v74; // [rsp+18h] [rbp-F8h]
  __int64 v75; // [rsp+20h] [rbp-F0h]
  char v76; // [rsp+20h] [rbp-F0h]
  _BYTE *v77; // [rsp+20h] [rbp-F0h]
  unsigned int v78; // [rsp+20h] [rbp-F0h]
  unsigned int v79; // [rsp+20h] [rbp-F0h]
  unsigned int v80; // [rsp+20h] [rbp-F0h]
  __int64 v81; // [rsp+20h] [rbp-F0h]
  unsigned int v82; // [rsp+20h] [rbp-F0h]
  __int64 v83; // [rsp+20h] [rbp-F0h]
  __int64 v84; // [rsp+28h] [rbp-E8h]
  __int64 v85; // [rsp+28h] [rbp-E8h]
  __int64 v86; // [rsp+28h] [rbp-E8h]
  __int64 v87; // [rsp+28h] [rbp-E8h]
  __int64 v88; // [rsp+28h] [rbp-E8h]
  __int64 v89; // [rsp+28h] [rbp-E8h]
  __int64 v90; // [rsp+28h] [rbp-E8h]
  __int64 v91; // [rsp+28h] [rbp-E8h]
  __int64 v92; // [rsp+28h] [rbp-E8h]
  __int64 v93; // [rsp+28h] [rbp-E8h]
  __int64 v94; // [rsp+28h] [rbp-E8h]
  __int64 v95; // [rsp+28h] [rbp-E8h]
  __int64 v96; // [rsp+28h] [rbp-E8h]
  __int64 v97; // [rsp+30h] [rbp-E0h] BYREF
  unsigned int v98; // [rsp+38h] [rbp-D8h]
  __int64 v99; // [rsp+40h] [rbp-D0h]
  unsigned int v100; // [rsp+48h] [rbp-C8h]
  int v101; // [rsp+50h] [rbp-C0h] BYREF
  __int64 v102; // [rsp+58h] [rbp-B8h]
  unsigned int v103; // [rsp+60h] [rbp-B0h]
  __int64 v104; // [rsp+68h] [rbp-A8h]
  unsigned int v105; // [rsp+70h] [rbp-A0h]
  __int64 v106; // [rsp+80h] [rbp-90h] BYREF
  unsigned int v107; // [rsp+88h] [rbp-88h]
  __int64 v108; // [rsp+90h] [rbp-80h]
  unsigned int v109; // [rsp+98h] [rbp-78h]
  __int64 v110; // [rsp+B0h] [rbp-60h] BYREF
  unsigned int v111; // [rsp+B8h] [rbp-58h]
  __int64 v112; // [rsp+C0h] [rbp-50h]
  unsigned int v113; // [rsp+C8h] [rbp-48h]

  v8 = *(_DWORD *)(a5 + 24);
  v9 = *(_QWORD *)(a5 + 8);
  if ( v8 )
  {
    v10 = (v8 - 1) & (((unsigned int)a3 >> 9) ^ ((unsigned int)a3 >> 4));
    v11 = (unsigned int *)(v9 + 48LL * v10);
    v12 = *(_QWORD *)v11;
    if ( *(_QWORD *)v11 == a3 )
    {
LABEL_3:
      if ( v11 != (unsigned int *)(v9 + 48LL * v8) )
      {
        *a1 = 0;
        sub_13E8810(a1, v11 + 2);
        return a1;
      }
    }
    else
    {
      v14 = 1;
      while ( v12 != -8 )
      {
        v34 = v14 + 1;
        v10 = (v8 - 1) & (v14 + v10);
        v11 = (unsigned int *)(v9 + 48LL * v10);
        v12 = *(_QWORD *)v11;
        if ( *(_QWORD *)v11 == a3 )
          goto LABEL_3;
        v14 = v34;
      }
    }
  }
  v15 = *(_BYTE *)(a3 + 16);
  if ( v15 <= 0x17u )
    goto LABEL_18;
  if ( v15 != 75 )
  {
    if ( (unsigned int)v15 - 35 <= 0x11 )
    {
      if ( a4 )
      {
        if ( v15 == 50 )
        {
LABEL_13:
          v16 = *(_QWORD *)(a3 - 48);
          if ( v16 != a3 && *(_QWORD *)(a3 - 24) != a3 )
          {
            v75 = a5;
            sub_13EDBA0(&v110, a2, *(_QWORD *)(a3 - 24));
            sub_13EDBA0(&v106, a2, v16);
            sub_13EA210(&v101, (__int64)&v106, (__int64)&v110);
            sub_13EA000((__int64)&v106);
            sub_13EA000((__int64)&v110);
            a5 = v75;
LABEL_16:
            v9 = *(_QWORD *)(a5 + 8);
            v8 = *(_DWORD *)(a5 + 24);
            goto LABEL_19;
          }
        }
      }
      else if ( v15 == 51 )
      {
        goto LABEL_13;
      }
    }
LABEL_18:
    v101 = 4;
    goto LABEL_19;
  }
  v20 = *(_BYTE **)(a3 - 24);
  v21 = *(_QWORD *)(a3 - 48);
  v22 = v20[16];
  v23 = *(_WORD *)(a3 + 18) & 0x7FFF;
  if ( v22 > 0x10u || (unsigned int)(v23 - 32) > 1 )
  {
    if ( *(_BYTE *)(*(_QWORD *)a2 + 8LL) != 11 )
      goto LABEL_18;
    if ( a2 == v21 )
      goto LABEL_64;
LABEL_31:
    v24 = *(_BYTE *)(v21 + 16);
    if ( v24 == 35 )
    {
      v57 = *(_QWORD *)(v21 - 48);
      if ( a2 == v57 && v57 && *(_BYTE *)(*(_QWORD *)(v21 - 24) + 16LL) == 13 )
      {
LABEL_35:
        if ( a2 != v21 )
        {
          v27 = *(_BYTE *)(v21 + 16);
          if ( v27 == 35 )
          {
            v58 = *(_QWORD *)(v21 - 48);
            if ( a2 != v58 )
              goto LABEL_39;
            if ( !v58 )
              goto LABEL_39;
            v87 = *(_QWORD *)(v21 - 24);
            if ( *(_BYTE *)(v87 + 16) != 13 )
              goto LABEL_39;
          }
          else if ( v27 != 5
                 || *(_WORD *)(v21 + 18) != 11
                 || (v60 = *(_QWORD *)(v21 - 24LL * (*(_DWORD *)(v21 + 20) & 0xFFFFFFF)), a2 != v60)
                 || !v60
                 || (v87 = *(_QWORD *)(v21 + 24 * (1LL - (*(_DWORD *)(v21 + 20) & 0xFFFFFFF))),
                     *(_BYTE *)(v87 + 16) != 13) )
          {
LABEL_39:
            v101 = 4;
            v9 = *(_QWORD *)(a5 + 8);
            v8 = *(_DWORD *)(a5 + 24);
            goto LABEL_19;
          }
LABEL_65:
          v63 = a5;
          v64 = a4;
          v70 = v23;
          v77 = v20;
          sub_15897D0(&v97, *(_DWORD *)(*(_QWORD *)v20 + 8LL) >> 8, 1);
          v45 = v70;
          v46 = v64;
          v47 = v63;
          v48 = v77[16];
          if ( v48 == 13 )
          {
            v107 = *((_DWORD *)v77 + 8);
            if ( v107 > 0x40 )
            {
              sub_16A4FD0(&v106, v77 + 24);
              v47 = v63;
              v46 = v64;
              v45 = v70;
            }
            else
            {
              v106 = *((_QWORD *)v77 + 3);
            }
            v65 = v47;
            v71 = v46;
            v78 = v45;
            sub_1589870(&v110, &v106);
            v45 = v78;
            v46 = v71;
            v47 = v65;
            if ( v98 > 0x40 && v97 )
            {
              j_j___libc_free_0_0(v97);
              v47 = v65;
              v46 = v71;
              v45 = v78;
            }
            v97 = v110;
            v49 = v111;
            v111 = 0;
            v98 = v49;
            if ( v100 > 0x40 && v99 )
            {
              v66 = v47;
              v72 = v46;
              v79 = v45;
              j_j___libc_free_0_0(v99);
              v45 = v79;
              v46 = v72;
              v99 = v112;
              v47 = v66;
              v100 = v113;
              if ( v111 > 0x40 && v110 )
              {
                j_j___libc_free_0_0(v110);
                v47 = v66;
                v46 = v72;
                v45 = v79;
              }
            }
            else
            {
              v99 = v112;
              v100 = v113;
            }
            if ( v107 > 0x40 && v106 )
            {
              v67 = v47;
              v73 = v46;
              v80 = v45;
              j_j___libc_free_0_0(v106);
              v45 = v80;
              v46 = v73;
              v47 = v67;
            }
          }
          else if ( v48 > 0x17u && (*((_QWORD *)v77 + 6) || *((__int16 *)v77 + 9) < 0) )
          {
            v54 = sub_1625790(v77, 4);
            v45 = v70;
            v46 = v64;
            v47 = v63;
            if ( v54 )
            {
              sub_1593050(&v110, v54);
              v45 = v70;
              v46 = v64;
              v47 = v63;
              if ( v98 > 0x40 && v97 )
              {
                j_j___libc_free_0_0(v97);
                v47 = v63;
                v46 = v64;
                v45 = v70;
              }
              v97 = v110;
              v55 = v111;
              v111 = 0;
              v98 = v55;
              if ( v100 > 0x40 && v99 )
              {
                v68 = v47;
                v74 = v46;
                v82 = v45;
                j_j___libc_free_0_0(v99);
                v45 = v82;
                v46 = v74;
                v99 = v112;
                v47 = v68;
                v100 = v113;
                if ( v111 > 0x40 && v110 )
                {
                  j_j___libc_free_0_0(v110);
                  v47 = v68;
                  v46 = v74;
                  v45 = v82;
                }
              }
              else
              {
                v99 = v112;
                v100 = v113;
              }
            }
          }
          if ( !v46 )
          {
            v83 = v47;
            v56 = sub_15FF0F0(v45);
            v47 = v83;
            v45 = v56;
          }
          v81 = v47;
          sub_158AE10(&v106, v45, &v97);
          v50 = v81;
          if ( v87 )
          {
            sub_158BC30(&v110, &v106, v87 + 24);
            v50 = v81;
            if ( v107 > 0x40 && v106 )
            {
              j_j___libc_free_0_0(v106);
              v50 = v81;
            }
            v106 = v110;
            v51 = v111;
            v111 = 0;
            v107 = v51;
            if ( v109 > 0x40 && v108 )
            {
              v88 = v50;
              j_j___libc_free_0_0(v108);
              v52 = v112;
              v53 = v113;
              v50 = v88;
              v108 = v112;
              v109 = v113;
              if ( v111 > 0x40 && v110 )
              {
                j_j___libc_free_0_0(v110);
                v53 = v109;
                v52 = v108;
                v50 = v88;
              }
            }
            else
            {
              v52 = v112;
              v53 = v113;
              v108 = v112;
            }
          }
          else
          {
            v53 = v109;
            v52 = v108;
          }
          v113 = v53;
          v89 = v50;
          v111 = v107;
          v107 = 0;
          v110 = v106;
          v112 = v52;
          v109 = 0;
          sub_13EA060(&v101, &v110);
          a5 = v89;
          if ( v113 > 0x40 && v112 )
          {
            j_j___libc_free_0_0(v112);
            a5 = v89;
          }
          if ( v111 > 0x40 && v110 )
          {
            v90 = a5;
            j_j___libc_free_0_0(v110);
            a5 = v90;
          }
          if ( v109 > 0x40 && v108 )
          {
            v91 = a5;
            j_j___libc_free_0_0(v108);
            a5 = v91;
          }
          if ( v107 > 0x40 && v106 )
          {
            v92 = a5;
            j_j___libc_free_0_0(v106);
            a5 = v92;
          }
          if ( v100 > 0x40 && v99 )
          {
            v93 = a5;
            j_j___libc_free_0_0(v99);
            a5 = v93;
          }
          if ( v98 > 0x40 && v97 )
          {
            v94 = a5;
            j_j___libc_free_0_0(v97);
            a5 = v94;
          }
          goto LABEL_16;
        }
LABEL_64:
        v87 = 0;
        goto LABEL_65;
      }
    }
    else if ( v24 == 5 && *(_WORD *)(v21 + 18) == 11 )
    {
      v59 = *(_QWORD *)(v21 - 24LL * (*(_DWORD *)(v21 + 20) & 0xFFFFFFF));
      if ( a2 == v59
        && v59
        && *(_BYTE *)(*(_QWORD *)(v21 + 24 * (1LL - (*(_DWORD *)(v21 + 20) & 0xFFFFFFF))) + 16LL) == 13 )
      {
        goto LABEL_35;
      }
    }
    v69 = a5;
    v76 = a4;
    v84 = *(_QWORD *)(a3 - 24);
    v25 = sub_15FF5D0(*(_WORD *)(a3 + 18) & 0x7FFF);
    a5 = v69;
    v23 = v25;
    a4 = v76;
    v26 = (_BYTE *)v21;
    v21 = v84;
    v20 = v26;
    goto LABEL_35;
  }
  if ( a2 != v21 )
  {
    if ( *(_BYTE *)(*(_QWORD *)a2 + 8LL) != 11 )
      goto LABEL_18;
    goto LABEL_31;
  }
  v101 = 0;
  if ( a4 == (v23 == 32) )
  {
    if ( v22 != 9 )
    {
      v96 = a5;
      sub_13EA740(&v101, (__int64)v20);
      a5 = v96;
      v9 = *(_QWORD *)(v96 + 8);
      v8 = *(_DWORD *)(v96 + 24);
    }
  }
  else if ( v22 != 9 )
  {
    v95 = a5;
    sub_13EA940(&v101, (__int64)v20);
    a5 = v95;
    v9 = *(_QWORD *)(v95 + 8);
    v8 = *(_DWORD *)(v95 + 24);
  }
LABEL_19:
  if ( !v8 )
  {
    ++*(_QWORD *)a5;
    goto LABEL_41;
  }
  v17 = (v8 - 1) & (((unsigned int)a3 >> 9) ^ ((unsigned int)a3 >> 4));
  v18 = (__int64 *)(v9 + 48LL * v17);
  v19 = *v18;
  if ( *v18 != a3 )
  {
    v35 = 1;
    v36 = 0;
    while ( v19 != -8 )
    {
      if ( !v36 && v19 == -16 )
        v36 = v18;
      v17 = (v8 - 1) & (v35 + v17);
      v18 = (__int64 *)(v9 + 48LL * v17);
      v19 = *v18;
      if ( *v18 == a3 )
        goto LABEL_21;
      ++v35;
    }
    v37 = *(_DWORD *)(a5 + 16);
    if ( v36 )
      v18 = v36;
    ++*(_QWORD *)a5;
    v32 = v37 + 1;
    if ( 4 * v32 < 3 * v8 )
    {
      if ( v8 - (v32 + *(_DWORD *)(a5 + 20)) > v8 >> 3 )
      {
LABEL_43:
        *(_DWORD *)(a5 + 16) = v32;
        if ( *v18 != -8 )
          --*(_DWORD *)(a5 + 20);
        *v18 = a3;
        *((_DWORD *)v18 + 2) = 0;
        goto LABEL_21;
      }
      v86 = a5;
      sub_13ED980(a5, v8);
      a5 = v86;
      v38 = *(_DWORD *)(v86 + 24);
      if ( v38 )
      {
        v39 = v38 - 1;
        v40 = *(_QWORD *)(v86 + 8);
        v41 = 1;
        v42 = 0;
        v43 = (v38 - 1) & (((unsigned int)a3 >> 9) ^ ((unsigned int)a3 >> 4));
        v18 = (__int64 *)(v40 + 48LL * v43);
        v32 = *(_DWORD *)(v86 + 16) + 1;
        v44 = *v18;
        if ( *v18 != a3 )
        {
          while ( v44 != -8 )
          {
            if ( !v42 && v44 == -16 )
              v42 = v18;
            v43 = v39 & (v41 + v43);
            v18 = (__int64 *)(v40 + 48LL * v43);
            v44 = *v18;
            if ( *v18 == a3 )
              goto LABEL_43;
            ++v41;
          }
          if ( v42 )
            v18 = v42;
        }
        goto LABEL_43;
      }
LABEL_168:
      ++*(_DWORD *)(a5 + 16);
      BUG();
    }
LABEL_41:
    v85 = a5;
    sub_13ED980(a5, 2 * v8);
    a5 = v85;
    v28 = *(_DWORD *)(v85 + 24);
    if ( v28 )
    {
      v29 = v28 - 1;
      v30 = *(_QWORD *)(v85 + 8);
      v31 = (v28 - 1) & (((unsigned int)a3 >> 9) ^ ((unsigned int)a3 >> 4));
      v18 = (__int64 *)(v30 + 48LL * v31);
      v32 = *(_DWORD *)(v85 + 16) + 1;
      v33 = *v18;
      if ( *v18 != a3 )
      {
        v61 = 1;
        v62 = 0;
        while ( v33 != -8 )
        {
          if ( !v62 && v33 == -16 )
            v62 = v18;
          v31 = v29 & (v61 + v31);
          v18 = (__int64 *)(v30 + 48LL * v31);
          v33 = *v18;
          if ( *v18 == a3 )
            goto LABEL_43;
          ++v61;
        }
        if ( v62 )
          v18 = v62;
      }
      goto LABEL_43;
    }
    goto LABEL_168;
  }
LABEL_21:
  sub_13E8810((int *)v18 + 2, (unsigned int *)&v101);
  *a1 = 0;
  sub_13E8810(a1, (unsigned int *)&v101);
  if ( v101 == 3 )
  {
    if ( v105 > 0x40 && v104 )
      j_j___libc_free_0_0(v104);
    if ( v103 > 0x40 && v102 )
      j_j___libc_free_0_0(v102);
  }
  return a1;
}
