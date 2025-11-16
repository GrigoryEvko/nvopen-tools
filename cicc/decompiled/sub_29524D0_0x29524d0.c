// Function: sub_29524D0
// Address: 0x29524d0
//
void __fastcall sub_29524D0(__int64 *a1, __int64 a2, __int64 a3)
{
  __int64 v3; // r14
  __int64 v4; // rax
  __int64 v5; // rsi
  __int64 v6; // rax
  int v7; // ecx
  __int64 v8; // rdi
  __int64 v9; // rcx
  __int64 v10; // rdx
  __int64 *v11; // rax
  __int64 v12; // r8
  __int64 v13; // rdx
  __int64 v14; // rcx
  unsigned __int64 v15; // r12
  unsigned __int64 v16; // rbx
  __int64 v17; // rax
  __int64 v18; // rdx
  __int64 v19; // r13
  unsigned int v20; // r14d
  bool v21; // al
  __int64 v22; // r15
  __int64 v23; // r14
  __int64 v24; // rax
  __int64 v25; // rdx
  unsigned __int64 v26; // rax
  unsigned __int64 v27; // rsi
  unsigned int v28; // r15d
  __int64 v29; // r9
  __int64 v30; // rax
  __int64 v31; // rax
  unsigned __int8 v32; // dl
  unsigned __int64 v33; // rbx
  int v34; // eax
  __int64 v35; // rbx
  unsigned int *v36; // r12
  unsigned int *v37; // r13
  __int64 v38; // rdx
  unsigned int v39; // esi
  bool v40; // zf
  unsigned __int64 v41; // rdx
  __int64 v42; // rax
  unsigned __int64 v43; // rax
  _BYTE *v44; // rax
  __int64 v45; // rax
  int v46; // eax
  unsigned int *v47; // r12
  unsigned int *v48; // r13
  __int64 v49; // rdx
  unsigned int v50; // esi
  __int64 v51; // r12
  int v52; // r13d
  __int64 v53; // rax
  __int64 v54; // rsi
  _QWORD *v55; // rax
  _QWORD *v56; // rdx
  __int64 v57; // r12
  __int64 v58; // rax
  __int64 v59; // rdx
  __int64 v60; // rax
  __int64 v61; // r13
  unsigned __int8 *v62; // rbx
  __int64 v63; // r15
  int v64; // eax
  unsigned __int8 *v65; // rdx
  int v66; // eax
  __int64 v67; // rax
  __int64 v68; // rdx
  __int64 v69; // rcx
  __int64 v70; // rsi
  __int64 v71; // rsi
  __int64 v72; // rax
  __int64 v73; // rcx
  __int64 v74; // rcx
  __int64 v75; // r13
  unsigned __int8 *v76; // rax
  int v77; // ebx
  const char *v78; // rax
  int v79; // r9d
  __int64 v80; // rcx
  unsigned __int64 v81; // [rsp+8h] [rbp-1A8h]
  __int64 v82; // [rsp+10h] [rbp-1A0h]
  char v83; // [rsp+1Fh] [rbp-191h]
  __int64 v86; // [rsp+38h] [rbp-178h]
  __int64 v87; // [rsp+40h] [rbp-170h]
  char v88; // [rsp+48h] [rbp-168h]
  __int64 v89; // [rsp+48h] [rbp-168h]
  __int64 v90; // [rsp+48h] [rbp-168h]
  __int64 v91; // [rsp+48h] [rbp-168h]
  unsigned __int64 v92; // [rsp+48h] [rbp-168h]
  __int64 v93; // [rsp+48h] [rbp-168h]
  __int64 v94; // [rsp+50h] [rbp-160h]
  __int64 v95; // [rsp+58h] [rbp-158h]
  unsigned __int8 **v96; // [rsp+60h] [rbp-150h]
  __int64 v97; // [rsp+68h] [rbp-148h]
  __int64 v98; // [rsp+70h] [rbp-140h]
  int v99; // [rsp+78h] [rbp-138h]
  unsigned int v100; // [rsp+7Ch] [rbp-134h]
  unsigned __int64 v101; // [rsp+80h] [rbp-130h] BYREF
  unsigned int v102; // [rsp+88h] [rbp-128h]
  _BYTE *v103[4]; // [rsp+90h] [rbp-120h] BYREF
  __int16 v104; // [rsp+B0h] [rbp-100h]
  const char *v105; // [rsp+C0h] [rbp-F0h] BYREF
  __int64 v106; // [rsp+C8h] [rbp-E8h]
  __int16 v107; // [rsp+E0h] [rbp-D0h]
  unsigned int *v108; // [rsp+F0h] [rbp-C0h] BYREF
  int v109; // [rsp+F8h] [rbp-B8h]
  char v110; // [rsp+100h] [rbp-B0h] BYREF
  __int64 v111; // [rsp+128h] [rbp-88h]
  __int64 v112; // [rsp+130h] [rbp-80h]
  _QWORD *v113; // [rsp+138h] [rbp-78h]
  __int64 v114; // [rsp+140h] [rbp-70h]
  __int64 v115; // [rsp+148h] [rbp-68h]
  void *v116; // [rsp+170h] [rbp-40h]

  v3 = a2;
  sub_23D0AB0((__int64)&v108, a2, 0, 0, 0);
  v4 = sub_AE4570(*a1, *(_QWORD *)(a2 + 8));
  v5 = *(_QWORD *)(a2 + 40);
  v87 = v4;
  v95 = *(_QWORD *)(v3 - 32LL * (*(_DWORD *)(v3 + 4) & 0x7FFFFFF));
  v6 = a1[2];
  v7 = *(_DWORD *)(v6 + 24);
  v8 = *(_QWORD *)(v6 + 8);
  if ( v7 )
  {
    v9 = (unsigned int)(v7 - 1);
    v10 = (unsigned int)v9 & (((unsigned int)v5 >> 9) ^ ((unsigned int)v5 >> 4));
    v11 = (__int64 *)(v8 + 16 * v10);
    v12 = *v11;
    if ( v5 == *v11 )
    {
LABEL_3:
      v82 = v11[1];
      if ( v82 )
      {
        v83 = sub_D48480(v82, v95, v10, v9);
        if ( v83 )
        {
          v51 = *(_QWORD *)(v95 + 16);
          if ( v51 )
          {
            v52 = 0;
            while ( 1 )
            {
              v53 = *(_QWORD *)(v51 + 24);
              if ( *(_BYTE *)v53 > 0x1Cu )
              {
                v54 = *(_QWORD *)(v53 + 40);
                if ( *(_BYTE *)(v82 + 84) )
                {
                  v55 = *(_QWORD **)(v82 + 64);
                  v56 = &v55[*(unsigned int *)(v82 + 76)];
                  if ( v55 != v56 )
                  {
                    while ( v54 != *v55 )
                    {
                      if ( v56 == ++v55 )
                        goto LABEL_94;
                    }
LABEL_92:
                    if ( v52 == 1 )
                      goto LABEL_75;
                    v52 = 1;
                  }
                }
                else if ( sub_C8CA60(v82 + 56, v54) )
                {
                  goto LABEL_92;
                }
              }
LABEL_94:
              v51 = *(_QWORD *)(v51 + 8);
              if ( !v51 )
                goto LABEL_5;
            }
          }
        }
      }
      else
      {
LABEL_75:
        v83 = 0;
      }
      goto LABEL_5;
    }
    v46 = 1;
    while ( v12 != -4096 )
    {
      v79 = v46 + 1;
      v10 = (unsigned int)v9 & (v46 + (_DWORD)v10);
      v11 = (__int64 *)(v8 + 16LL * (unsigned int)v10);
      v12 = *v11;
      if ( v5 == *v11 )
        goto LABEL_3;
      v46 = v79;
    }
  }
  v82 = 0;
  v83 = 0;
LABEL_5:
  if ( (*(_BYTE *)(v3 + 7) & 0x40) != 0 )
    v13 = *(_QWORD *)(v3 - 8);
  else
    v13 = v3 - 32LL * (*(_DWORD *)(v3 + 4) & 0x7FFFFFF);
  v96 = (unsigned __int8 **)(v13 + 32);
  v98 = sub_BB5290(v3) & 0xFFFFFFFFFFFFFFF9LL | 4;
  v99 = *(_DWORD *)(v3 + 4) & 0x7FFFFFF;
  if ( v99 == 1 )
  {
    v94 = 0;
    goto LABEL_36;
  }
  v100 = 1;
  v94 = 0;
  v86 = v3;
  do
  {
    v15 = v98 & 0xFFFFFFFFFFFFFFF8LL;
    v16 = v98 & 0xFFFFFFFFFFFFFFF8LL;
    v17 = v100;
    v14 = v100 + 1;
    v18 = (v98 >> 1) & 3;
    ++v100;
    v97 = v18;
    if ( ((v98 >> 1) & 3) == 0 )
      goto LABEL_43;
    v18 = *(_DWORD *)(v86 + 4) & 0x7FFFFFF;
    v19 = *(_QWORD *)(v86 + 32 * (v17 - v18));
    if ( *(_BYTE *)v19 != 17
      || ((v20 = *(_DWORD *)(v19 + 32), v20 <= 0x40)
        ? (v21 = *(_QWORD *)(v19 + 24) == 0)
        : (v21 = v20 == (unsigned int)sub_C444A0(v19 + 24)),
          !v21) )
    {
      v22 = *a1;
      if ( v98 )
      {
        if ( v97 == 2 )
        {
          v23 = v98 & 0xFFFFFFFFFFFFFFF8LL;
          if ( v15 )
          {
LABEL_17:
            v88 = sub_AE5020(v22, v23);
            v24 = sub_9208B0(v22, v23);
            v106 = v25;
            v26 = ((1LL << v88) + ((unsigned __int64)(v24 + 7) >> 3) - 1) >> v88 << v88;
            goto LABEL_18;
          }
LABEL_67:
          v23 = sub_BCBAE0(v15, *v96, v18);
          goto LABEL_17;
        }
        if ( (_DWORD)v97 != 1 )
          goto LABEL_67;
        if ( v15 )
          v23 = *(_QWORD *)(v15 + 24);
        else
          v23 = sub_BCBAE0(0, *v96, v18);
      }
      else
      {
        v23 = sub_BCBAE0(v15, *v96, v18);
        if ( v97 != 1 )
          goto LABEL_17;
      }
      v42 = sub_9208B0(v22, v23);
      v106 = v25;
      v26 = (unsigned __int64)(v42 + 7) >> 3;
LABEL_18:
      v105 = (const char *)v26;
      LOBYTE(v106) = v25;
      v27 = sub_CA1930(&v105);
      v28 = *(_DWORD *)(v87 + 8) >> 8;
      v102 = v28;
      if ( v28 > 0x40 )
      {
        sub_C43690((__int64)&v101, v27, 0);
        v28 = v102;
        if ( v102 > 0x40 )
        {
          if ( v28 - (unsigned int)sub_C444A0((__int64)&v101) <= 0x40 && *(_QWORD *)v101 == 1 )
            goto LABEL_24;
          if ( (unsigned int)sub_C44630((__int64)&v101) == 1 )
          {
            v104 = 257;
            v34 = sub_C444A0((__int64)&v101);
            goto LABEL_53;
          }
LABEL_22:
          v104 = 257;
          v89 = sub_AD8D80(v87, (__int64)&v101);
          v29 = (*(__int64 (__fastcall **)(__int64, __int64, __int64, __int64, _QWORD, _QWORD))(*(_QWORD *)v114 + 32LL))(
                  v114,
                  17,
                  v19,
                  v89,
                  0,
                  0);
          if ( v29 )
            goto LABEL_23;
          v107 = 257;
          v93 = sub_B504D0(17, v19, v89, (__int64)&v105, 0, 0);
          (*(void (__fastcall **)(__int64, __int64, _BYTE **, __int64, __int64))(*(_QWORD *)v115 + 16LL))(
            v115,
            v93,
            v103,
            v111,
            v112);
          v29 = v93;
          if ( v108 == &v108[4 * v109] )
            goto LABEL_23;
          v92 = v98 & 0xFFFFFFFFFFFFFFF8LL;
          v35 = v29;
          v81 = v98 & 0xFFFFFFFFFFFFFFF8LL;
          v47 = v108;
          v48 = &v108[4 * v109];
          do
          {
            v49 = *((_QWORD *)v47 + 1);
            v50 = *v47;
            v47 += 4;
            sub_B99FD0(v35, v50, v49);
          }
          while ( v48 != v47 );
LABEL_57:
          v29 = v35;
          v15 = v81;
          v16 = v92;
LABEL_23:
          v19 = v29;
LABEL_24:
          v105 = "uglygep";
          v107 = 259;
          v103[0] = (_BYTE *)v19;
          v30 = sub_BCB2B0(v113);
          v31 = sub_921130(&v108, v30, v95, v103, 1, (__int64)&v105, 0);
          v14 = v94;
          v95 = v31;
          if ( !v94 )
            v14 = v31;
          v94 = v14;
          if ( v102 > 0x40 && v101 )
            j_j___libc_free_0_0(v101);
          goto LABEL_29;
        }
      }
      else
      {
        v101 = v27;
      }
      if ( v101 == 1 )
        goto LABEL_24;
      if ( v101 && (v101 & (v101 - 1)) == 0 )
      {
        _BitScanReverse64(&v43, v101);
        v104 = 257;
        v34 = v28 + (v43 ^ 0x3F) - 64;
LABEL_53:
        v90 = sub_AD64C0(v87, v28 - 1 - v34, 0);
        v29 = (*(__int64 (__fastcall **)(__int64, __int64, __int64, __int64, _QWORD, _QWORD))(*(_QWORD *)v114 + 32LL))(
                v114,
                25,
                v19,
                v90,
                0,
                0);
        if ( v29 )
          goto LABEL_23;
        v107 = 257;
        v91 = sub_B504D0(25, v19, v90, (__int64)&v105, 0, 0);
        (*(void (__fastcall **)(__int64, __int64, _BYTE **, __int64, __int64))(*(_QWORD *)v115 + 16LL))(
          v115,
          v91,
          v103,
          v111,
          v112);
        v29 = v91;
        if ( v108 == &v108[4 * v109] )
          goto LABEL_23;
        v92 = v98 & 0xFFFFFFFFFFFFFFF8LL;
        v35 = v29;
        v81 = v98 & 0xFFFFFFFFFFFFFFF8LL;
        v36 = v108;
        v37 = &v108[4 * v109];
        do
        {
          v38 = *((_QWORD *)v36 + 1);
          v39 = *v36;
          v36 += 4;
          sub_B99FD0(v35, v39, v38);
        }
        while ( v37 != v36 );
        goto LABEL_57;
      }
      goto LABEL_22;
    }
LABEL_29:
    if ( v98 )
    {
      if ( v97 == 2 )
      {
        if ( v15 )
          goto LABEL_32;
      }
      else if ( v97 == 1 && v15 )
      {
        v16 = *(_QWORD *)(v15 + 24);
LABEL_32:
        v32 = *(_BYTE *)(v16 + 8);
        if ( v32 == 16 )
          goto LABEL_33;
        goto LABEL_44;
      }
    }
LABEL_43:
    v16 = sub_BCBAE0(v15, *v96, v18);
    v32 = *(_BYTE *)(v16 + 8);
    if ( v32 == 16 )
    {
LABEL_33:
      v98 = *(_QWORD *)(v16 + 24) & 0xFFFFFFFFFFFFFFF9LL | 4;
      goto LABEL_34;
    }
LABEL_44:
    v33 = v16 & 0xFFFFFFFFFFFFFFF9LL;
    if ( (unsigned int)v32 - 17 > 1 )
    {
      v40 = v32 == 15;
      v41 = 0;
      if ( v40 )
        v41 = v33;
      v98 = v41;
    }
    else
    {
      v98 = v33 | 2;
    }
LABEL_34:
    v96 += 4;
  }
  while ( v100 != v99 );
  v3 = v86;
LABEL_36:
  if ( a3 )
  {
    v44 = (_BYTE *)sub_AD64C0(v87, a3, 0);
    v105 = "uglygep";
    v107 = 259;
    v103[0] = v44;
    v45 = sub_BCB2B0(v113);
    v95 = sub_921130(&v108, v45, v95, v103, 1, (__int64)&v105, 0);
  }
  else
  {
    v83 = 0;
  }
  if ( v94 && *(_BYTE *)v94 == 63 )
  {
    v57 = 0;
    if ( *(_BYTE *)v95 == 63 )
      v57 = v95;
    if ( v83 )
    {
      v58 = *(_QWORD *)(v94 + 16);
      if ( v58 )
      {
        if ( !*(_QWORD *)(v58 + 8) )
        {
          if ( v57 )
          {
            if ( *(_QWORD *)(v94 + 40) == *(_QWORD *)(v57 + 40) && v57 != v94 )
            {
              v59 = *(_DWORD *)(v94 + 4) & 0x7FFFFFF;
              v60 = *(_DWORD *)(v57 + 4) & 0x7FFFFFF;
              if ( (_DWORD)v60 == (_DWORD)v59 && (_DWORD)v59 == 2 )
              {
                v61 = *(_QWORD *)(v94 - 64);
                v62 = *(unsigned __int8 **)(v94 - 32);
                v63 = *(_QWORD *)(v57 - 32 * v60);
                if ( !(unsigned __int8)sub_D48480(v82, (__int64)v62, v59, v14)
                  && *(_QWORD *)(v61 + 8) == *(_QWORD *)(v63 + 8) )
                {
                  if ( (v64 = *v62, (unsigned __int8)v64 <= 0x1Cu)
                    || (unsigned int)(v64 - 54) <= 2
                    && ((v62[7] & 0x40) == 0
                      ? (v65 = &v62[-32 * (*((_DWORD *)v62 + 1) & 0x7FFFFFF)])
                      : (v65 = (unsigned __int8 *)*((_QWORD *)v62 - 1)),
                        **((_BYTE **)v65 + 4) == 17 && (v62 = *(unsigned __int8 **)v65, **(_BYTE **)v65 <= 0x1Cu))
                    || (v66 = *v62, (unsigned int)(v66 - 42) > 0x11)
                    || (((_BYTE)v66 - 42) & 0xFD) != 0
                    || **((_BYTE **)v62 - 8) != 17 && **((_BYTE **)v62 - 4) != 17 )
                  {
                    v67 = v94 + 32 * (1LL - (*(_DWORD *)(v94 + 4) & 0x7FFFFFF));
                    v68 = *(_QWORD *)v67;
                    v69 = *(_QWORD *)(v57 + 32 * (1LL - (*(_DWORD *)(v57 + 4) & 0x7FFFFFF)));
                    if ( v69 )
                    {
                      if ( v68 )
                      {
                        v70 = *(_QWORD *)(v67 + 8);
                        **(_QWORD **)(v67 + 16) = v70;
                        if ( v70 )
                          *(_QWORD *)(v70 + 16) = *(_QWORD *)(v67 + 16);
                      }
                      *(_QWORD *)v67 = v69;
                      v71 = *(_QWORD *)(v69 + 16);
                      *(_QWORD *)(v67 + 8) = v71;
                      if ( v71 )
                        *(_QWORD *)(v71 + 16) = v67 + 8;
                      *(_QWORD *)(v67 + 16) = v69 + 16;
                      *(_QWORD *)(v69 + 16) = v67;
                      v72 = v57 + 32 * (1LL - (*(_DWORD *)(v57 + 4) & 0x7FFFFFF));
                      if ( *(_QWORD *)v72 )
                      {
                        v73 = *(_QWORD *)(v72 + 8);
                        **(_QWORD **)(v72 + 16) = v73;
                        if ( v73 )
                          goto LABEL_127;
                      }
LABEL_128:
                      *(_QWORD *)v72 = v68;
                      if ( v68 )
                        goto LABEL_129;
                    }
                    else if ( v68 )
                    {
                      v80 = *(_QWORD *)(v67 + 8);
                      **(_QWORD **)(v67 + 16) = v80;
                      if ( v80 )
                        *(_QWORD *)(v80 + 16) = *(_QWORD *)(v67 + 16);
                      *(_QWORD *)v67 = 0;
                      v72 = v57 + 32 * (1LL - (*(_DWORD *)(v57 + 4) & 0x7FFFFFF));
                      if ( *(_QWORD *)v72 )
                      {
                        v73 = *(_QWORD *)(v72 + 8);
                        **(_QWORD **)(v72 + 16) = v73;
                        if ( v73 )
                        {
LABEL_127:
                          *(_QWORD *)(v73 + 16) = *(_QWORD *)(v72 + 16);
                          goto LABEL_128;
                        }
                      }
                      *(_QWORD *)v72 = v68;
LABEL_129:
                      v74 = *(_QWORD *)(v68 + 16);
                      *(_QWORD *)(v72 + 8) = v74;
                      if ( v74 )
                        *(_QWORD *)(v74 + 16) = v72 + 8;
                      *(_QWORD *)(v72 + 16) = v68 + 16;
                      *(_QWORD *)(v68 + 16) = v72;
                    }
                    v75 = sub_B43CC0(v94);
                    LODWORD(v106) = sub_AE2980(v75, *(_DWORD *)(*(_QWORD *)(v94 + 8) + 8LL) >> 8)[3];
                    if ( (unsigned int)v106 > 0x40 )
                      sub_C43690((__int64)&v105, 0, 0);
                    else
                      v105 = 0;
                    v76 = sub_BD45C0((unsigned __int8 *)v94, v75, (__int64)&v105, 0, 0, 0, 0, 0);
                    if ( !(unsigned __int8)sub_D62CA0((__int64)v76, v103, v75, a1[3], 0, 0) )
                      goto LABEL_145;
                    v77 = v106;
                    if ( (unsigned int)v106 <= 0x40 )
                    {
                      v78 = v105;
                      goto LABEL_137;
                    }
                    if ( v77 - (unsigned int)sub_C444A0((__int64)&v105) > 0x40 )
                      goto LABEL_145;
                    v78 = *(const char **)v105;
LABEL_137:
                    if ( v103[0] >= v78 )
                    {
                      sub_B4DE00(v94, 1);
                    }
                    else
                    {
LABEL_145:
                      sub_B4DDE0(v94, 0);
                      sub_B4DDE0(v57, 0);
                    }
                    if ( (unsigned int)v106 > 0x40 && v105 )
                      j_j___libc_free_0_0((unsigned __int64)v105);
                  }
                }
              }
            }
          }
        }
      }
    }
  }
  sub_BD84D0(v3, v95);
  sub_B43D60((_QWORD *)v3);
  nullsub_61();
  v116 = &unk_49DA100;
  nullsub_63();
  if ( v108 != (unsigned int *)&v110 )
    _libc_free((unsigned __int64)v108);
}
