// Function: sub_34AA450
// Address: 0x34aa450
//
__int64 __fastcall sub_34AA450(
        __int64 a1,
        __int64 a2,
        __int64 a3,
        __int64 a4,
        unsigned __int64 a5,
        unsigned __int64 a6,
        __int64 a7)
{
  __int64 v7; // r14
  __int64 v8; // rdx
  __int64 *v9; // r15
  __int64 v10; // rbx
  _QWORD *v11; // rax
  _QWORD *v12; // rdx
  __int64 v13; // rcx
  __int64 v14; // rsi
  __int64 v15; // rdi
  int v16; // edx
  unsigned __int64 *v17; // rax
  __int64 v18; // rdx
  __int64 v19; // rdx
  __int64 v20; // r12
  __int64 v21; // rdx
  __int64 v22; // rcx
  __int64 v23; // r8
  __int64 v24; // r9
  _QWORD *v25; // rax
  __int64 v26; // rcx
  unsigned __int64 i; // rdx
  unsigned int v28; // ebx
  unsigned int *v29; // r14
  __int64 v30; // rdx
  __int64 v31; // rcx
  __int64 v32; // r8
  __int64 v33; // r9
  __int64 v34; // rdx
  __int64 v35; // rcx
  unsigned __int64 v36; // r8
  unsigned __int64 v37; // r9
  __int64 v39; // rax
  __int64 v40; // rsi
  __int64 v41; // rdx
  __int64 v42; // rcx
  unsigned __int64 v43; // r8
  unsigned __int64 v44; // r9
  __int64 v45; // rdx
  __int64 v46; // rcx
  __int64 v47; // r12
  unsigned __int64 *v48; // r14
  unsigned int v49; // esi
  unsigned __int64 v50; // r12
  __int64 v51; // r8
  unsigned int *v52; // rax
  unsigned int *v53; // rax
  unsigned __int64 v54; // r8
  unsigned __int64 v55; // rsi
  __int64 v56; // rax
  int v57; // eax
  __int64 v58; // r8
  unsigned int v59; // r14d
  __int64 v60; // rax
  unsigned __int64 v61; // r12
  __int64 v62; // rax
  __int64 v63; // rax
  __int64 v64; // r8
  __int64 v65; // rdx
  __int64 v66; // rcx
  __int64 v67; // r8
  __int64 v68; // r9
  __int64 v69; // rax
  unsigned __int64 v70; // rsi
  unsigned __int64 v71; // rdx
  unsigned __int64 v72; // r15
  unsigned __int64 *v73; // rax
  __int64 v74; // rax
  unsigned __int64 *v75; // rax
  unsigned int v76; // esi
  __int64 v77; // [rsp+8h] [rbp-338h]
  unsigned __int64 *v79; // [rsp+18h] [rbp-328h]
  __int64 v82; // [rsp+30h] [rbp-310h]
  unsigned __int64 v83; // [rsp+48h] [rbp-2F8h]
  unsigned __int64 v84; // [rsp+48h] [rbp-2F8h]
  unsigned __int64 v85; // [rsp+48h] [rbp-2F8h]
  int v86; // [rsp+50h] [rbp-2F0h]
  __int64 *v87; // [rsp+58h] [rbp-2E8h]
  unsigned __int8 v89; // [rsp+68h] [rbp-2D8h]
  unsigned __int64 *v90; // [rsp+68h] [rbp-2D8h]
  __int64 v91; // [rsp+68h] [rbp-2D8h]
  __int64 v92; // [rsp+70h] [rbp-2D0h] BYREF
  _BYTE *v93; // [rsp+78h] [rbp-2C8h] BYREF
  __int64 v94; // [rsp+80h] [rbp-2C0h]
  _BYTE v95[64]; // [rsp+88h] [rbp-2B8h] BYREF
  unsigned int v96; // [rsp+C8h] [rbp-278h]
  unsigned __int64 v97; // [rsp+D0h] [rbp-270h]
  unsigned __int64 v98; // [rsp+D8h] [rbp-268h]
  unsigned int *v99; // [rsp+E0h] [rbp-260h] BYREF
  _BYTE *v100; // [rsp+E8h] [rbp-258h] BYREF
  __int64 v101; // [rsp+F0h] [rbp-250h]
  _BYTE v102[64]; // [rsp+F8h] [rbp-248h] BYREF
  int v103; // [rsp+138h] [rbp-208h]
  __int64 v104; // [rsp+140h] [rbp-200h]
  __int64 v105; // [rsp+148h] [rbp-1F8h]
  __int64 v106; // [rsp+150h] [rbp-1F0h] BYREF
  unsigned int v107[48]; // [rsp+158h] [rbp-1E8h] BYREF
  __int64 v108; // [rsp+218h] [rbp-128h]
  __int64 v109; // [rsp+220h] [rbp-120h]
  unsigned __int64 *v110; // [rsp+230h] [rbp-110h] BYREF
  _QWORD v111[33]; // [rsp+238h] [rbp-108h] BYREF

  v7 = a6;
  v79 = (unsigned __int64 *)(a1 + 376);
  v106 = a1 + 376;
  v109 = a1 + 376;
  memset(v107, 0, sizeof(v107));
  v8 = *(unsigned int *)(a2 + 72);
  v77 = a5;
  v9 = *(__int64 **)(a2 + 64);
  v108 = 0;
  v87 = &v9[v8];
  v86 = 0;
  while ( v87 != v9 )
  {
    v10 = *v9;
    if ( *(_BYTE *)(v7 + 28) )
    {
      v11 = *(_QWORD **)(v7 + 8);
      v12 = &v11[*(unsigned int *)(v7 + 20)];
      if ( v11 == v12 )
        goto LABEL_16;
      while ( v10 != *v11 )
      {
        if ( v12 == ++v11 )
          goto LABEL_16;
      }
    }
    else if ( !sub_C8CA60(v7, *v9) )
    {
      goto LABEL_16;
    }
    v89 = *(_BYTE *)(a3 + 8);
    v13 = v89;
    v14 = v89 & 1;
    if ( (v89 & 1) != 0 )
    {
      v15 = a3 + 16;
      v16 = 3;
    }
    else
    {
      v15 = *(_QWORD *)(a3 + 16);
      v39 = *(unsigned int *)(a3 + 24);
      if ( !(_DWORD)v39 )
        goto LABEL_51;
      v16 = v39 - 1;
    }
    v13 = v16 & (((unsigned int)v10 >> 9) ^ ((unsigned int)v10 >> 4));
    v17 = (unsigned __int64 *)(v15 + 16 * v13);
    a5 = *v17;
    if ( v10 == *v17 )
      goto LABEL_10;
    v57 = 1;
    while ( a5 != -4096 )
    {
      a6 = (unsigned int)(v57 + 1);
      v13 = v16 & (unsigned int)(v57 + v13);
      v17 = (unsigned __int64 *)(v15 + 16LL * (unsigned int)v13);
      a5 = *v17;
      if ( v10 == *v17 )
        goto LABEL_10;
      v57 = a6;
    }
    if ( (_BYTE)v14 )
    {
      v56 = 64;
      goto LABEL_52;
    }
    v39 = *(unsigned int *)(a3 + 24);
LABEL_51:
    v56 = 16 * v39;
LABEL_52:
    v17 = (unsigned __int64 *)(v15 + v56);
LABEL_10:
    v18 = 64;
    if ( !(_BYTE)v14 )
    {
      v13 = a3;
      v18 = 16LL * *(unsigned int *)(a3 + 24);
    }
    v19 = v15 + v18;
    if ( v17 == (unsigned __int64 *)v19 )
    {
      v28 = 0;
      goto LABEL_25;
    }
    v20 = v17[1];
    if ( v86 )
    {
      v40 = v17[1];
      v110 = &v111[1];
      v111[0] = 0x800000000LL;
      sub_34A5A20((__int64)&v106, v40, (__int64)&v110, v13, a5, a6);
      sub_34A2530(v107, v40, v41, v42, v43, v44);
      v47 = 2LL * LODWORD(v111[0]);
      v90 = &v110[v47];
      if ( v110 == &v110[v47] )
      {
LABEL_41:
        if ( v90 != &v111[1] )
          _libc_free((unsigned __int64)v90);
        goto LABEL_15;
      }
      v82 = v7;
      v48 = v110;
      while ( 2 )
      {
        v49 = v108;
        v50 = *v48;
        v51 = v48[1];
        if ( (_DWORD)v108 )
        {
          v55 = *v48;
          v84 = v48[1];
          v99 = v107;
          v100 = v102;
          v101 = 0x400000000LL;
          sub_34A3C90((__int64)&v99, v55, v45, v46, v51, a6);
          v54 = v84;
        }
        else
        {
          if ( HIDWORD(v108) != 11 )
          {
            if ( HIDWORD(v108) )
            {
              v52 = &v107[2];
              do
              {
                if ( v50 <= *(_QWORD *)v52 )
                  break;
                ++v49;
                v52 += 4;
              }
              while ( HIDWORD(v108) != v49 );
            }
            LODWORD(v99) = v49;
            HIDWORD(v108) = sub_34A32D0((__int64)v107, (unsigned int *)&v99, HIDWORD(v108), v50, v51, 0);
            goto LABEL_39;
          }
          v99 = v107;
          v101 = 0x400000000LL;
          v53 = &v107[2];
          v100 = v102;
          do
          {
            if ( v50 <= *(_QWORD *)v53 )
              break;
            ++v49;
            v53 += 4;
          }
          while ( v49 != 11 );
          v83 = v51;
          sub_34A26E0((__int64)&v99, v49, HIDWORD(v108), v46, v51, a6);
          v54 = v83;
        }
        sub_34A8E00((__int64)&v99, v50, v54, 0);
        if ( v100 != v102 )
          _libc_free((unsigned __int64)v100);
LABEL_39:
        v48 += 2;
        if ( v90 == v48 )
        {
          v7 = v82;
          v90 = v110;
          goto LABEL_41;
        }
        continue;
      }
    }
    sub_34A2530(v107, v14, 0, v13, a5, a6);
    sub_34AA090((__int64)&v106, v20, v21, v22, v23, v24);
LABEL_15:
    ++v86;
LABEL_16:
    ++v9;
  }
  v110 = v79;
  v111[25] = v79;
  memset(v111, 0, 200);
  if ( *(_BYTE *)(a7 + 28) )
  {
    v25 = *(_QWORD **)(a7 + 8);
    v26 = a2;
    for ( i = (unsigned __int64)&v25[*(unsigned int *)(a7 + 20)]; (_QWORD *)i != v25; ++v25 )
    {
      if ( a2 == *v25 )
        goto LABEL_22;
    }
  }
  else if ( sub_C8CA60(a7, a2) )
  {
    goto LABEL_22;
  }
  v99 = v107;
  v100 = v102;
  v101 = 0x400000000LL;
  sub_34A26E0((__int64)&v99, 0, i, v26, a5, a6);
  v26 = (__int64)v99;
  i = (unsigned int)v101;
  v59 = v99[48];
  if ( v59 )
  {
    i = (unsigned int)v101;
    v69 = (unsigned int)(v101 - 1);
    if ( v59 > (unsigned int)v69 )
    {
      do
      {
        v70 = (unsigned __int64)v100;
        v71 = i + 1;
        v72 = *(_QWORD *)(*(_QWORD *)&v100[16 * v69] + 8LL * *(unsigned int *)&v100[16 * v69 + 12])
            & 0xFFFFFFFFFFFFFFC0LL;
        v58 = (*(_QWORD *)(*(_QWORD *)&v100[16 * v69] + 8LL * *(unsigned int *)&v100[16 * v69 + 12]) & 0x3FLL) + 1;
        if ( v71 > HIDWORD(v101) )
        {
          v91 = (*(_QWORD *)(*(_QWORD *)&v100[16 * v69] + 8LL * *(unsigned int *)&v100[16 * v69 + 12]) & 0x3FLL) + 1;
          sub_C8D5F0((__int64)&v100, v102, v71, 0x10u, v58, a6);
          v70 = (unsigned __int64)v100;
          v58 = v91;
        }
        v73 = (unsigned __int64 *)(v70 + 16LL * (unsigned int)v101);
        *v73 = v72;
        v73[1] = v58;
        v69 = (unsigned int)v101;
        i = (unsigned int)(v101 + 1);
        LODWORD(v101) = v101 + 1;
      }
      while ( v59 > (unsigned int)v69 );
      v26 = (__int64)v99;
    }
  }
  v92 = v26;
  v93 = v95;
  v94 = 0x400000000LL;
  if ( !(_DWORD)i )
  {
    v97 = 0;
    v98 = 0;
LABEL_61:
    v96 = -1;
    goto LABEL_62;
  }
  sub_349DB40((__int64)&v93, (__int64)&v100, i, v26, v58, a6);
  v96 = 0;
  v97 = 0;
  v98 = 0;
  if ( !(_DWORD)v94 )
    goto LABEL_61;
  v26 = *((unsigned int *)v93 + 2);
  if ( *((_DWORD *)v93 + 3) >= (unsigned int)v26 )
    goto LABEL_61;
  v97 = *(_QWORD *)sub_34A2590((__int64)&v92);
  v98 = *(_QWORD *)sub_34A25B0((__int64)&v92);
LABEL_62:
  if ( v100 != v102 )
    _libc_free((unsigned __int64)v100);
  a5 = v97;
  v101 = 0x400000000LL;
  v60 = v96;
  v100 = v102;
  v99 = 0;
  v104 = 0;
  v105 = 0;
  v103 = -1;
LABEL_67:
  if ( (_DWORD)v60 == -1 )
    goto LABEL_72;
  while ( 1 )
  {
    while ( 1 )
    {
      v61 = v60 + a5;
      v62 = sub_349D6E0(v77, __ROL8__(v60 + a5, 32));
      v63 = sub_B10CD0(*(_QWORD *)(v62 + 48) + 56LL);
      if ( !(unsigned __int8)sub_35068F0(a1 + 112, v63, a2) )
        sub_34A8ED0((__int64)v111, v61, v61, 0, v64, a6);
      a5 = v97;
      i = v97 + v96;
      if ( i < v98 )
      {
        v60 = ++v96;
        goto LABEL_67;
      }
      v74 = (__int64)&v93[16 * (unsigned int)v94 - 16];
      i = (unsigned int)(*(_DWORD *)(v74 + 12) + 1);
      *(_DWORD *)(v74 + 12) = i;
      v26 = (unsigned int)v94;
      if ( (_DWORD)i == *(_DWORD *)&v93[16 * (unsigned int)v94 - 8] )
      {
        v76 = *(_DWORD *)(v92 + 192);
        if ( v76 )
        {
          sub_F03D40((__int64 *)&v93, v76);
          v26 = (unsigned int)v94;
        }
      }
      if ( !(_DWORD)v26 )
        break;
      v26 = *((unsigned int *)v93 + 2);
      if ( *((_DWORD *)v93 + 3) >= (unsigned int)v26 )
        break;
      v96 = 0;
      v97 = *(_QWORD *)sub_34A2590((__int64)&v92);
      v85 = v97;
      v75 = (unsigned __int64 *)sub_34A25B0((__int64)&v92);
      a5 = v85;
      v98 = *v75;
      v60 = 0;
    }
    v96 = -1;
    a5 = 0;
    v97 = 0;
    v98 = 0;
LABEL_72:
    if ( !(v98 | a5) )
      break;
    v60 = 0xFFFFFFFFLL;
  }
  if ( v93 != v95 )
    _libc_free((unsigned __int64)v93);
LABEL_22:
  v28 = 0;
  sub_34A9020((__int64)&v106, (__int64)&v110, i, v26, a5, a6);
  v14 = (__int64)&v106;
  v29 = (unsigned int *)sub_34A3910(a1, a2, a4);
  if ( !(unsigned __int8)sub_34A2770((__int64)v29, (__int64)&v106, v30, v31, v32, v33) )
  {
    v28 = 1;
    sub_34A2530(v29 + 2, (__int64)&v106, v34, v35, v36, v37);
    v14 = (__int64)&v106;
    sub_34AA090((__int64)v29, (__int64)&v106, v65, v66, v67, v68);
  }
  sub_34A2530((unsigned int *)v111, (__int64)&v106, v34, v35, v36, v37);
LABEL_25:
  sub_34A2530(v107, v14, v19, v13, a5, a6);
  return v28;
}
