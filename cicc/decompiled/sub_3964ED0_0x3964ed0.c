// Function: sub_3964ED0
// Address: 0x3964ed0
//
__int64 __fastcall sub_3964ED0(__int64 a1, __int64 *a2, __int64 *a3, __int64 *a4, __int64 a5)
{
  __int64 *v5; // rbx
  unsigned __int64 v6; // rdi
  __int64 v8; // r13
  __int64 v9; // rax
  __int64 v10; // rbx
  unsigned __int8 v11; // r14
  _QWORD *v12; // rsi
  _QWORD *v13; // rdx
  unsigned __int8 v14; // al
  __int64 v15; // r15
  __int64 v16; // r14
  __int64 v17; // r13
  unsigned __int64 v18; // rdi
  _QWORD *v19; // rax
  _BYTE *v20; // rbx
  __int64 **v21; // r12
  __int64 **v22; // rbx
  __int64 *v23; // rax
  unsigned __int64 v24; // rdi
  _BYTE *v25; // rax
  __int64 v26; // r15
  __int64 v27; // rdx
  int v28; // eax
  __int64 v29; // rsi
  int v30; // edx
  int v31; // edi
  unsigned int v32; // eax
  __int64 v33; // rcx
  __int64 v34; // r14
  __int64 v35; // rax
  __int64 v36; // rax
  __int64 *v37; // rbx
  __int64 *v38; // rax
  __int64 *v39; // r14
  __int64 v40; // r12
  __int64 v41; // r13
  __int64 v42; // rdx
  __int64 v43; // rcx
  unsigned int v44; // edi
  __int64 *v45; // rax
  __int64 v46; // rsi
  __int64 v47; // r11
  unsigned int v48; // esi
  unsigned int v49; // r8d
  __int64 v50; // rdi
  unsigned int v51; // edx
  __int64 v52; // rax
  __int64 v53; // rcx
  unsigned int v54; // eax
  unsigned __int64 v55; // rdx
  __int64 v56; // rax
  __int64 v57; // rax
  __int64 v58; // rdx
  int v59; // eax
  __int64 v60; // rax
  __int64 v61; // r9
  int i; // r10d
  int v63; // r10d
  __int64 v64; // r9
  int v65; // eax
  int v66; // edx
  __int64 v67; // rcx
  int v68; // r9d
  _BYTE *v69; // rdx
  int v70; // r9d
  __int64 *v71; // r10
  unsigned int v72; // eax
  __int64 v73; // [rsp+0h] [rbp-2F0h]
  unsigned int v74; // [rsp+8h] [rbp-2E8h]
  __int64 v76; // [rsp+38h] [rbp-2B8h]
  __int64 v77; // [rsp+40h] [rbp-2B0h]
  unsigned __int8 v78; // [rsp+52h] [rbp-29Eh]
  unsigned __int8 v79; // [rsp+53h] [rbp-29Dh]
  __int64 v80; // [rsp+58h] [rbp-298h]
  __int64 v81; // [rsp+60h] [rbp-290h]
  __int64 v82; // [rsp+68h] [rbp-288h]
  __int64 v83; // [rsp+70h] [rbp-280h]
  __int64 v84; // [rsp+70h] [rbp-280h]
  __int64 *v86; // [rsp+80h] [rbp-270h]
  __int64 *v87; // [rsp+80h] [rbp-270h]
  int v88; // [rsp+80h] [rbp-270h]
  __int64 v90[2]; // [rsp+90h] [rbp-260h] BYREF
  __int64 v91[2]; // [rsp+A0h] [rbp-250h] BYREF
  __int16 v92; // [rsp+B0h] [rbp-240h]
  _BYTE *v93; // [rsp+D0h] [rbp-220h] BYREF
  __int64 v94; // [rsp+D8h] [rbp-218h]
  _BYTE v95[64]; // [rsp+E0h] [rbp-210h] BYREF
  __int64 v96; // [rsp+120h] [rbp-1D0h] BYREF
  _BYTE *v97; // [rsp+128h] [rbp-1C8h]
  _BYTE *v98; // [rsp+130h] [rbp-1C0h]
  __int64 v99; // [rsp+138h] [rbp-1B8h]
  int v100; // [rsp+140h] [rbp-1B0h]
  _BYTE v101[72]; // [rsp+148h] [rbp-1A8h] BYREF
  int v102[6]; // [rsp+190h] [rbp-160h] BYREF
  char v103[8]; // [rsp+1A8h] [rbp-148h] BYREF
  __int64 v104; // [rsp+1B0h] [rbp-140h]
  unsigned __int64 v105; // [rsp+1B8h] [rbp-138h]
  __int64 v106; // [rsp+210h] [rbp-E0h] BYREF
  _BYTE *v107; // [rsp+218h] [rbp-D8h]
  _BYTE *v108; // [rsp+220h] [rbp-D0h]
  __int64 v109; // [rsp+228h] [rbp-C8h]
  int v110; // [rsp+230h] [rbp-C0h]
  _BYTE v111[184]; // [rsp+238h] [rbp-B8h] BYREF

  v76 = *a2;
  if ( *(_BYTE *)(*a2 + 16) <= 0x17u )
    BUG();
  v5 = a3;
  v82 = *a3;
  if ( *a3 == *(_QWORD *)(*a2 + 40) )
    return 0;
  sub_3963E30((__int64)v102, a1, a2, *a3, 0);
  v79 = sub_3961900(v102, a1 + 64);
  if ( !v79 )
    goto LABEL_4;
  v8 = *v5;
  v78 = *((_BYTE *)v5 + 193);
  if ( v78 )
  {
    v84 = sub_157ED20(v8);
    goto LABEL_31;
  }
  v106 = 0;
  v107 = v111;
  v108 = v111;
  v9 = *a2;
  v109 = 16;
  v110 = 0;
  if ( !*(_QWORD *)(v9 + 8) )
    goto LABEL_19;
  v86 = v5;
  v10 = *(_QWORD *)(v9 + 8);
  v11 = 0;
  do
  {
    while ( 1 )
    {
      v13 = sub_1648700(v10);
      v14 = *((_BYTE *)v13 + 16);
      if ( v14 <= 0x17u )
        goto LABEL_13;
      if ( v14 == 77 )
        break;
      if ( v8 == v13[5] )
        goto LABEL_17;
LABEL_13:
      v10 = *(_QWORD *)(v10 + 8);
      if ( !v10 )
        goto LABEL_18;
    }
    if ( (*((_BYTE *)v13 + 23) & 0x40) != 0 )
      v12 = (_QWORD *)*(v13 - 1);
    else
      v12 = &v13[-3 * (*((_DWORD *)v13 + 5) & 0xFFFFFFF)];
    v11 = v79;
    if ( v8 != v12[3 * *((unsigned int *)v13 + 14) + 1 + -1431655765 * (unsigned int)((v10 - (__int64)v12) >> 3)] )
      goto LABEL_13;
LABEL_17:
    sub_165A590((__int64)&v96, (__int64)&v106, (__int64)v13);
    v10 = *(_QWORD *)(v10 + 8);
  }
  while ( v10 );
LABEL_18:
  v78 = v11;
  v5 = v86;
LABEL_19:
  v15 = *(_QWORD *)(v8 + 48);
  v16 = v8 + 40;
  if ( v15 != v8 + 40 )
  {
    v83 = v8;
    v87 = v5;
    while ( 1 )
    {
      if ( !v15 )
        BUG();
      v17 = v15 - 24;
      if ( *(_BYTE *)(v15 - 8) != 77 )
        break;
LABEL_62:
      v15 = *(_QWORD *)(v15 + 8);
      if ( v16 == v15 )
      {
        v8 = v83;
        v5 = v87;
        goto LABEL_64;
      }
    }
    v18 = (unsigned __int64)v108;
    v19 = v107;
    if ( v108 == v107 )
    {
      v20 = &v108[8 * HIDWORD(v109)];
      if ( v108 == v20 )
      {
        v69 = v108;
      }
      else
      {
        do
        {
          if ( v17 == *v19 )
            break;
          ++v19;
        }
        while ( v20 != (_BYTE *)v19 );
        v69 = &v108[8 * HIDWORD(v109)];
      }
    }
    else
    {
      v20 = &v108[8 * (unsigned int)v109];
      v19 = sub_16CC9F0((__int64)&v106, v15 - 24);
      if ( v17 == *v19 )
      {
        v18 = (unsigned __int64)v108;
        if ( v108 == v107 )
          v69 = &v108[8 * HIDWORD(v109)];
        else
          v69 = &v108[8 * (unsigned int)v109];
      }
      else
      {
        v18 = (unsigned __int64)v108;
        if ( v108 != v107 )
        {
          v19 = &v108[8 * (unsigned int)v109];
          goto LABEL_27;
        }
        v19 = &v108[8 * HIDWORD(v109)];
        v69 = v19;
      }
    }
    while ( v69 != (_BYTE *)v19 && *v19 >= 0xFFFFFFFFFFFFFFFELL )
      ++v19;
LABEL_27:
    if ( v19 != (_QWORD *)v20 )
    {
      v84 = v15 - 24;
      v5 = v87;
      goto LABEL_29;
    }
    goto LABEL_62;
  }
LABEL_64:
  v60 = sub_157ED20(v8);
  v18 = (unsigned __int64)v108;
  v84 = v60;
LABEL_29:
  if ( (_BYTE *)v18 != v107 )
    _libc_free(v18);
LABEL_31:
  v96 = 0;
  v93 = v95;
  v94 = 0x800000000LL;
  v97 = v101;
  v98 = v101;
  v99 = 8;
  v100 = 0;
  sub_3962830(v76, (__int64)v103, (__int64)&v93, (__int64)&v96);
  v106 = 0;
  v107 = v111;
  v108 = v111;
  v109 = 8;
  v110 = 0;
  sub_1953970((__int64)v91, (__int64)&v106, v82);
  v21 = (__int64 **)v5[18];
  v22 = &v21[*((unsigned int *)v5 + 38)];
  if ( v21 != v22 )
  {
    do
    {
      v23 = *v21++;
      sub_1412190((__int64)&v106, *v23);
      v24 = (unsigned __int64)v108;
      v25 = v107;
    }
    while ( v22 != v21 );
    v88 = v94 - 1;
    if ( (int)v94 - 1 >= 0 )
    {
      v78 = v79;
LABEL_35:
      v81 = 0;
      v80 = 8LL * v88;
      while ( 1 )
      {
        v26 = *(_QWORD *)&v93[v80];
        v27 = *(_QWORD *)(a1 + 56);
        v28 = *(_DWORD *)(v27 + 80);
        if ( v28 )
        {
          v29 = *(_QWORD *)(v27 + 64);
          v30 = v28 - 1;
          v31 = 1;
          v32 = (v28 - 1) & (((unsigned int)v26 >> 9) ^ ((unsigned int)v26 >> 4));
          v33 = *(_QWORD *)(v29 + 16LL * v32);
          if ( v26 == v33 )
          {
LABEL_38:
            sub_1412190(a5, *(_QWORD *)&v93[v80]);
          }
          else
          {
            while ( v33 != -8 )
            {
              v32 = v30 & (v31 + v32);
              v33 = *(_QWORD *)(v29 + 16LL * v32);
              if ( v26 == v33 )
                goto LABEL_38;
              ++v31;
            }
          }
        }
        v34 = sub_15F4880(v26);
        v35 = v81;
        if ( v76 == v26 )
          v35 = v34;
        v81 = v35;
        v36 = 3LL * (*(_DWORD *)(v34 + 20) & 0xFFFFFFF);
        if ( (*(_BYTE *)(v34 + 23) & 0x40) != 0 )
        {
          v37 = *(__int64 **)(v34 - 8);
          v38 = &v37[v36];
        }
        else
        {
          v37 = (__int64 *)(v34 - v36 * 8);
          v38 = (__int64 *)v34;
        }
        if ( v37 != v38 )
          break;
LABEL_55:
        v90[0] = (__int64)sub_1649960(v26);
        v92 = 773;
        v90[1] = v58;
        v91[0] = (__int64)v90;
        v91[1] = (__int64)".remat";
        sub_164B780(v34, v91);
        sub_15F2120(v34, v84);
        sub_3964770(a1, v26, v34, (__int64)&v106);
        --v88;
        v80 -= 8;
        if ( v88 == -1 )
          goto LABEL_91;
        v84 = v34;
      }
      v77 = v34;
      v39 = v38;
      while ( 1 )
      {
        v40 = *v37;
        if ( !(unsigned __int8)sub_3960EF0((_BYTE *)*v37) )
        {
LABEL_45:
          v37 += 3;
          if ( v39 == v37 )
            goto LABEL_54;
          continue;
        }
        v41 = *(_QWORD *)(a1 + 56);
        v42 = *(unsigned int *)(v41 + 136);
        v43 = *(_QWORD *)(v41 + 120);
        if ( !(_DWORD)v42 )
          goto LABEL_61;
        v44 = (v42 - 1) & (((unsigned int)v82 >> 9) ^ ((unsigned int)v82 >> 4));
        v45 = (__int64 *)(v43 + 16LL * v44);
        v46 = *v45;
        if ( v82 != *v45 )
          break;
LABEL_49:
        v47 = v45[1];
        v90[0] = v40;
        v48 = *(_DWORD *)(v41 + 80);
        if ( v48 )
        {
          v49 = v48 - 1;
          v50 = *(_QWORD *)(v41 + 64);
          v51 = (v48 - 1) & (((unsigned int)v40 >> 9) ^ ((unsigned int)v40 >> 4));
          v52 = v50 + 16LL * v51;
          v53 = *(_QWORD *)v52;
          if ( v40 != *(_QWORD *)v52 )
          {
            v74 = (v48 - 1) & (((unsigned int)v40 >> 9) ^ ((unsigned int)v40 >> 4));
            v61 = *(_QWORD *)v52;
            for ( i = 1; ; ++i )
            {
              if ( v61 == -8 )
                goto LABEL_53;
              v74 = v49 & (v74 + i);
              v61 = *(_QWORD *)(v50 + 16LL * v74);
              if ( v40 == v61 )
                break;
            }
            v63 = 1;
            v64 = 0;
            while ( v53 != -8 )
            {
              if ( v53 != -16 || v64 )
                v52 = v64;
              v70 = v63 + 1;
              v51 = v49 & (v63 + v51);
              v71 = (__int64 *)(v50 + 16LL * v51);
              v53 = *v71;
              if ( v40 == *v71 )
              {
                v72 = *((_DWORD *)v71 + 2);
                v55 = v72 & 0x3F;
                v56 = 8LL * (v72 >> 6);
                goto LABEL_52;
              }
              v63 = v70;
              v64 = v52;
              v52 = v50 + 16LL * v51;
            }
            if ( !v64 )
              v64 = v52;
            v65 = *(_DWORD *)(v41 + 72);
            ++*(_QWORD *)(v41 + 56);
            v66 = v65 + 1;
            if ( 4 * (v65 + 1) >= 3 * v48 )
            {
              v73 = v47;
              v48 *= 2;
            }
            else
            {
              v67 = v40;
              if ( v48 - *(_DWORD *)(v41 + 76) - v66 > v48 >> 3 )
              {
LABEL_86:
                *(_DWORD *)(v41 + 72) = v66;
                if ( *(_QWORD *)v64 != -8 )
                  --*(_DWORD *)(v41 + 76);
                *(_QWORD *)v64 = v67;
                v55 = 0;
                v56 = 0;
                *(_DWORD *)(v64 + 8) = 0;
                goto LABEL_52;
              }
              v73 = v47;
            }
            sub_1BFE340(v41 + 56, v48);
            sub_1BFD9C0(v41 + 56, v90, v91);
            v64 = v91[0];
            v67 = v90[0];
            v47 = v73;
            v66 = *(_DWORD *)(v41 + 72) + 1;
            goto LABEL_86;
          }
          v54 = *(_DWORD *)(v52 + 8);
          v55 = v54 & 0x3F;
          v56 = 8LL * (v54 >> 6);
LABEL_52:
          v57 = *(_QWORD *)(*(_QWORD *)(v47 + 24) + v56);
          if ( _bittest64(&v57, v55) )
            goto LABEL_45;
        }
LABEL_53:
        v37 += 3;
        sub_39627B0(a1, v40, a5);
        if ( v39 == v37 )
        {
LABEL_54:
          v34 = v77;
          goto LABEL_55;
        }
      }
      v59 = 1;
      while ( v46 != -8 )
      {
        v68 = v59 + 1;
        v44 = (v42 - 1) & (v59 + v44);
        v45 = (__int64 *)(v43 + 16LL * v44);
        v46 = *v45;
        if ( v82 == *v45 )
          goto LABEL_49;
        v59 = v68;
      }
LABEL_61:
      v45 = (__int64 *)(v43 + 16 * v42);
      goto LABEL_49;
    }
    v81 = 0;
LABEL_101:
    *a4 = v81;
    if ( (_BYTE *)v24 == v25 )
      goto LABEL_94;
LABEL_93:
    _libc_free(v24);
    goto LABEL_94;
  }
  v88 = v94 - 1;
  if ( (int)v94 - 1 >= 0 )
    goto LABEL_35;
  v81 = 0;
LABEL_91:
  v24 = (unsigned __int64)v108;
  v25 = v107;
  if ( v78 )
    goto LABEL_101;
  if ( v108 != v107 )
    goto LABEL_93;
LABEL_94:
  if ( v98 != v97 )
    _libc_free((unsigned __int64)v98);
  if ( v93 != v95 )
  {
    _libc_free((unsigned __int64)v93);
    v6 = v105;
    if ( v105 == v104 )
      return v79;
    goto LABEL_5;
  }
LABEL_4:
  v6 = v105;
  if ( v105 != v104 )
LABEL_5:
    _libc_free(v6);
  return v79;
}
