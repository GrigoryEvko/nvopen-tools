// Function: sub_1984A30
// Address: 0x1984a30
//
__int64 __fastcall sub_1984A30(__int64 a1, __int64 a2, __int64 a3, __m128i a4, __m128i a5)
{
  __int64 v5; // r13
  _QWORD *v8; // r15
  int v9; // r8d
  int *v10; // r9
  unsigned __int8 v11; // al
  __int64 v12; // rax
  unsigned int v13; // edx
  __int64 *v14; // rax
  __int64 v15; // rcx
  int *v16; // rdx
  signed __int64 v17; // rcx
  int *v18; // rax
  int *v19; // r10
  __int64 v20; // rdi
  __int64 v21; // rsi
  int *v22; // r10
  __int64 v23; // rsi
  __int64 v24; // rax
  __int64 v25; // rax
  int *v26; // rax
  __int64 v27; // rdx
  _BOOL8 v28; // rdi
  __int64 v29; // rax
  __int64 v30; // r15
  __int64 *v31; // rax
  __int64 *v32; // rsi
  __int64 v33; // rcx
  __int64 v34; // rdx
  __int64 *v35; // rax
  __int64 i; // r15
  int v37; // ecx
  __int64 v38; // rdi
  __int64 v39; // rsi
  int *v40; // rax
  __int64 v41; // rcx
  __int64 v42; // rdx
  unsigned int v43; // eax
  __int64 v44; // rcx
  int v45; // r8d
  int v46; // r9d
  unsigned int v47; // r13d
  unsigned int v48; // eax
  _QWORD *v49; // r13
  __int64 v50; // rbx
  unsigned __int64 v51; // r12
  unsigned __int64 v52; // rdi
  unsigned __int64 v53; // rdi
  __int64 v55; // rax
  int *v56; // rdx
  __int64 *v57; // rax
  __int64 *v58; // rdi
  __int64 v59; // rdx
  unsigned int v60; // eax
  __int64 v61; // rcx
  int v62; // r8d
  int v63; // r9d
  int v64; // eax
  __int64 v65; // r14
  _QWORD *v66; // rbx
  __int64 v67; // rax
  unsigned __int64 v68; // r15
  __int64 v69; // rbx
  __int64 v70; // rdx
  __int64 *v71; // rsi
  unsigned int v72; // edi
  __int64 *v73; // rcx
  int *v74; // rdi
  int *v75; // r15
  __int64 v76; // rax
  int *v77; // rax
  __int64 v78; // rdx
  _BOOL8 v79; // rdi
  signed __int64 v80; // [rsp+0h] [rbp-15E0h]
  int *v81; // [rsp+8h] [rbp-15D8h]
  signed __int64 v82; // [rsp+10h] [rbp-15D0h]
  int *v83; // [rsp+10h] [rbp-15D0h]
  int *v84; // [rsp+10h] [rbp-15D0h]
  int v85; // [rsp+18h] [rbp-15C8h]
  __int64 v86; // [rsp+18h] [rbp-15C8h]
  int *v87; // [rsp+18h] [rbp-15C8h]
  int *v88; // [rsp+18h] [rbp-15C8h]
  __int64 v90; // [rsp+30h] [rbp-15B0h] BYREF
  int v91; // [rsp+38h] [rbp-15A8h] BYREF
  int *v92; // [rsp+40h] [rbp-15A0h]
  int *v93; // [rsp+48h] [rbp-1598h]
  int *v94; // [rsp+50h] [rbp-1590h]
  __int64 v95; // [rsp+58h] [rbp-1588h]
  __int64 v96; // [rsp+60h] [rbp-1580h] BYREF
  _BYTE *v97; // [rsp+68h] [rbp-1578h] BYREF
  __int64 v98; // [rsp+70h] [rbp-1570h]
  _BYTE v99[128]; // [rsp+78h] [rbp-1568h] BYREF
  __int64 v100; // [rsp+F8h] [rbp-14E8h] BYREF
  _BYTE *v101; // [rsp+100h] [rbp-14E0h]
  _BYTE *v102; // [rsp+108h] [rbp-14D8h]
  __int64 v103; // [rsp+110h] [rbp-14D0h]
  int v104; // [rsp+118h] [rbp-14C8h]
  _BYTE v105[128]; // [rsp+120h] [rbp-14C0h] BYREF
  _BYTE *v106; // [rsp+1A0h] [rbp-1440h] BYREF
  __int64 v107; // [rsp+1A8h] [rbp-1438h]
  _BYTE v108[5168]; // [rsp+1B0h] [rbp-1430h] BYREF

  v5 = *(_QWORD *)(a2 + 8);
  v106 = v108;
  v91 = 0;
  v92 = 0;
  v93 = &v91;
  v94 = &v91;
  v95 = 0;
  v107 = 0x1000000000LL;
  if ( !v5 )
    goto LABEL_97;
  do
  {
    v8 = sub_1648700(v5);
    if ( !(unsigned __int8)sub_1983A80((__int64)v8, *(_QWORD **)(a1 + 64)) )
    {
      v11 = *((_BYTE *)v8 + 16);
      if ( v11 <= 0x17u )
        goto LABEL_89;
      if ( (unsigned int)v11 - 35 > 0x11 )
      {
        if ( v11 == 56 )
        {
          v12 = v8[3 * ((*((_DWORD *)v8 + 5) & 0xFFFFFFFu) - 1 - (unsigned __int64)(*((_DWORD *)v8 + 5) & 0xFFFFFFF))];
          if ( !v12 )
LABEL_96:
            BUG();
LABEL_7:
          if ( *(_BYTE *)(v12 + 16) == 13 )
          {
            v13 = *(_DWORD *)(v12 + 32);
            v14 = *(__int64 **)(v12 + 24);
            if ( v13 <= 0x40 )
              v15 = (__int64)((_QWORD)v14 << (64 - (unsigned __int8)v13)) >> (64 - (unsigned __int8)v13);
            else
              v15 = *v14;
            v16 = v92;
            v17 = abs64(v15);
            if ( v92 )
            {
              v18 = v92;
              v19 = &v91;
              do
              {
                while ( 1 )
                {
                  v20 = *((_QWORD *)v18 + 2);
                  v21 = *((_QWORD *)v18 + 3);
                  if ( v17 <= *((_QWORD *)v18 + 4) )
                    break;
                  v18 = (int *)*((_QWORD *)v18 + 3);
                  if ( !v21 )
                    goto LABEL_15;
                }
                v19 = v18;
                v18 = (int *)*((_QWORD *)v18 + 2);
              }
              while ( v20 );
LABEL_15:
              if ( v19 != &v91 && v17 >= *((_QWORD *)v19 + 4) )
                goto LABEL_89;
              v22 = &v91;
              do
              {
                while ( 1 )
                {
                  v23 = *((_QWORD *)v16 + 2);
                  v24 = *((_QWORD *)v16 + 3);
                  if ( v17 <= *((_QWORD *)v16 + 4) )
                    break;
                  v16 = (int *)*((_QWORD *)v16 + 3);
                  if ( !v24 )
                    goto LABEL_21;
                }
                v22 = v16;
                v16 = (int *)*((_QWORD *)v16 + 2);
              }
              while ( v23 );
LABEL_21:
              if ( v22 != &v91 && v17 >= *((_QWORD *)v22 + 4) )
                goto LABEL_28;
            }
            else
            {
              v22 = &v91;
            }
            v81 = v22;
            v82 = v17;
            v25 = sub_22077B0(48);
            *(_QWORD *)(v25 + 40) = 0;
            *(_QWORD *)(v25 + 32) = v82;
            v80 = v82;
            v83 = (int *)v25;
            v26 = (int *)sub_1984930(&v90, v81, (__int64 *)(v25 + 32));
            if ( v27 )
            {
              v28 = v26 || &v91 == (int *)v27 || v80 < *(_QWORD *)(v27 + 32);
              sub_220F040(v28, v83, v27, &v91);
              ++v95;
              v22 = v83;
            }
            else
            {
              v74 = v83;
              v84 = v26;
              j_j___libc_free_0(v74, 48);
              v22 = v84;
            }
LABEL_28:
            *((_QWORD *)v22 + 5) = v8;
            goto LABEL_33;
          }
        }
      }
      else if ( (v11 & 0xEF) == 0x23 )
      {
        v12 = *(v8 - 3);
        if ( !v12 )
          goto LABEL_96;
        goto LABEL_7;
      }
      v29 = (unsigned int)v107;
      if ( (unsigned int)v107 >= HIDWORD(v107) )
      {
        sub_16CD150((__int64)&v106, v108, 0, 8, v9, (int)v10);
        v29 = (unsigned int)v107;
      }
      *(_QWORD *)&v106[8 * v29] = v8;
      LODWORD(v107) = v107 + 1;
      goto LABEL_33;
    }
    v55 = *(unsigned int *)(a1 + 5232);
    if ( (unsigned int)v55 >= *(_DWORD *)(a1 + 5236) )
    {
      sub_16CD150(a1 + 5224, (const void *)(a1 + 5240), 0, 8, v9, (int)v10);
      v55 = *(unsigned int *)(a1 + 5232);
    }
    *(_QWORD *)(*(_QWORD *)(a1 + 5224) + 8 * v55) = v8;
    ++*(_DWORD *)(a1 + 5232);
LABEL_33:
    v5 = *(_QWORD *)(v5 + 8);
  }
  while ( v5 );
  if ( !v95 )
    goto LABEL_89;
  if ( v95 == 1 )
  {
    if ( (_DWORD)v107 )
      goto LABEL_100;
    goto LABEL_89;
  }
  if ( !(_DWORD)v107 )
    goto LABEL_37;
LABEL_100:
  v56 = v92;
  if ( !v92 )
  {
    v75 = &v91;
    goto LABEL_150;
  }
  v57 = (__int64 *)v92;
  v58 = (__int64 *)&v91;
  do
  {
    if ( v57[4] < 0 )
    {
      v57 = (__int64 *)v57[3];
    }
    else
    {
      v58 = v57;
      v57 = (__int64 *)v57[2];
    }
  }
  while ( v57 );
  if ( v58 != (__int64 *)&v91 && v58[4] <= 0 )
  {
LABEL_89:
    if ( v106 != v108 )
    {
      _libc_free((unsigned __int64)v106);
      v47 = 0;
      goto LABEL_91;
    }
LABEL_97:
    v47 = 0;
    goto LABEL_91;
  }
  v75 = &v91;
  do
  {
    if ( *((__int64 *)v56 + 4) < 0 )
    {
      v56 = (int *)*((_QWORD *)v56 + 3);
    }
    else
    {
      v75 = v56;
      v56 = (int *)*((_QWORD *)v56 + 2);
    }
  }
  while ( v56 );
  if ( v75 == &v91 || *((__int64 *)v75 + 4) > 0 )
  {
LABEL_150:
    v87 = v75;
    v76 = sub_22077B0(48);
    *(_QWORD *)(v76 + 32) = 0;
    v75 = (int *)v76;
    *(_QWORD *)(v76 + 40) = 0;
    v77 = (int *)sub_1984930(&v90, v87, (__int64 *)(v76 + 32));
    if ( v78 )
    {
      v79 = v77 || &v91 == (int *)v78 || *(_QWORD *)(v78 + 32) > 0LL;
      sub_220F040(v79, v75, v78, &v91);
      ++v95;
    }
    else
    {
      v88 = v77;
      j_j___libc_free_0(v75, 48);
      v75 = v88;
    }
  }
  *((_QWORD *)v75 + 5) = a2;
  v85 = v107;
  if ( !(_DWORD)v107 )
LABEL_37:
    v85 = sub_1648EF0(*((_QWORD *)v93 + 5));
  v30 = (__int64)v93;
  if ( v93 != &v91 )
  {
    while ( !*(_QWORD *)(v30 + 32) || sub_1648CD0(*(_QWORD *)(v30 + 40), v85) )
    {
      v30 = sub_220EEE0(v30);
      if ( (int *)v30 == &v91 )
        goto LABEL_42;
    }
    goto LABEL_89;
  }
LABEL_42:
  if ( v106 != v108 )
    _libc_free((unsigned __int64)v106);
  v31 = (__int64 *)v92;
  if ( !v92 )
    goto LABEL_51;
  v32 = (__int64 *)&v91;
  do
  {
    while ( 1 )
    {
      v33 = v31[2];
      v34 = v31[3];
      if ( v31[4] >= 0 )
        break;
      v31 = (__int64 *)v31[3];
      if ( !v34 )
        goto LABEL_49;
    }
    v32 = v31;
    v31 = (__int64 *)v31[2];
  }
  while ( v33 );
LABEL_49:
  if ( v32 == (__int64 *)&v91 || v32[4] > 0 )
  {
LABEL_51:
    v35 = *(__int64 **)(a3 + 8);
    if ( *(__int64 **)(a3 + 16) != v35 )
      goto LABEL_52;
    v34 = *(unsigned int *)(a3 + 28);
    v71 = &v35[v34];
    v72 = *(_DWORD *)(a3 + 28);
    if ( v35 == v71 )
    {
LABEL_157:
      if ( v72 >= *(_DWORD *)(a3 + 24) )
      {
LABEL_52:
        sub_16CCBA0(a3, a2);
      }
      else
      {
        *(_DWORD *)(a3 + 28) = v72 + 1;
        *v71 = a2;
        ++*(_QWORD *)a3;
      }
    }
    else
    {
      v73 = 0;
      while ( 1 )
      {
        v34 = *v35;
        if ( a2 == *v35 )
          break;
        if ( v34 == -2 )
          v73 = v35;
        if ( v71 == ++v35 )
        {
          if ( !v73 )
            goto LABEL_157;
          *v73 = a2;
          --*(_DWORD *)(a3 + 32);
          ++*(_QWORD *)a3;
          break;
        }
      }
    }
  }
  i = (__int64)v93;
  v37 = 0;
  v100 = 0;
  v97 = v99;
  v101 = v105;
  v102 = v105;
  v98 = 0x1000000000LL;
  v103 = 16;
  v104 = 0;
  v96 = 0;
  v106 = v108;
  v107 = 0x1000000000LL;
  if ( v93 == &v91 )
  {
    v47 = 0;
    goto LABEL_85;
  }
LABEL_73:
  v96 = *(_QWORD *)(i + 40);
  sub_16CCD50((__int64)&v100, a3, v34, v37, v9, (int)v10);
  for ( i = sub_220EEE0(i); (int *)i != &v91; i = sub_220EEE0(i) )
  {
    if ( !v96 )
      goto LABEL_73;
    v38 = (unsigned int)v98;
    if ( (_DWORD)v98 )
    {
      v39 = *(_QWORD *)(i + 32) - 1LL;
      v40 = v92;
      if ( !v92 )
        goto LABEL_63;
      v10 = &v91;
      do
      {
        while ( 1 )
        {
          v41 = *((_QWORD *)v40 + 2);
          v42 = *((_QWORD *)v40 + 3);
          if ( v39 <= *((_QWORD *)v40 + 4) )
            break;
          v40 = (int *)*((_QWORD *)v40 + 3);
          if ( !v42 )
            goto LABEL_61;
        }
        v10 = v40;
        v40 = (int *)*((_QWORD *)v40 + 2);
      }
      while ( v41 );
LABEL_61:
      if ( v10 == &v91 || v39 < *((_QWORD *)v10 + 4) )
      {
LABEL_63:
        LOBYTE(v43) = sub_1984400(a1, (__int64)&v96, a4, a5);
        v47 = v43;
        if ( !(_BYTE)v43 )
          goto LABEL_113;
        v48 = v107;
        if ( (unsigned int)v107 >= HIDWORD(v107) )
        {
          sub_1984210((__int64 *)&v106, 0);
          v48 = v107;
        }
        v49 = &v106[320 * v48];
        if ( v49 )
        {
          *v49 = v96;
          v49[1] = v49 + 3;
          v49[2] = 0x1000000000LL;
          if ( (_DWORD)v98 )
            sub_1983D00((__int64)(v49 + 1), (__int64)&v97, v48, v44, v45, v46);
          sub_16CCCB0(v49 + 19, (__int64)(v49 + 24), (__int64)&v100);
          v48 = v107;
        }
        LODWORD(v98) = 0;
        LODWORD(v107) = v48 + 1;
        v96 = *(_QWORD *)(i + 40);
        continue;
      }
      if ( (unsigned int)v98 >= HIDWORD(v98) )
      {
LABEL_110:
        sub_16CD150((__int64)&v97, v99, 0, 8, v9, (int)v10);
        v38 = (unsigned int)v98;
      }
    }
    else
    {
      v9 = HIDWORD(v98);
      if ( !HIDWORD(v98) )
        goto LABEL_110;
    }
    *(_QWORD *)&v97[8 * v38] = *(_QWORD *)(i + 40);
    LODWORD(v98) = v98 + 1;
  }
  if ( (_DWORD)v98 )
  {
    LOBYTE(v60) = sub_1984400(a1, (__int64)&v96, a4, a5);
    v47 = v60;
    if ( (_BYTE)v60 )
    {
      v64 = v107;
      if ( (unsigned int)v107 >= HIDWORD(v107) )
      {
        sub_1984210((__int64 *)&v106, 0);
        v64 = v107;
      }
      v65 = (__int64)v106;
      v66 = &v106[320 * v64];
      if ( v66 )
      {
        *v66 = v96;
        v66[1] = v66 + 3;
        v66[2] = 0x1000000000LL;
        if ( (_DWORD)v98 )
          sub_1983D00((__int64)(v66 + 1), (__int64)&v97, (unsigned int)v98, v61, v62, v63);
        sub_16CCCB0(v66 + 19, (__int64)(v66 + 24), (__int64)&v100);
        v65 = (__int64)v106;
        v64 = v107;
      }
      v67 = (unsigned int)(v64 + 1);
      LODWORD(v107) = v67;
      v67 *= 320;
      v86 = v65 + v67;
      v68 = 0xCCCCCCCCCCCCCCCDLL * (v67 >> 6);
      v59 = *(unsigned int *)(a1 + 96);
      if ( v68 > (unsigned __int64)*(unsigned int *)(a1 + 100) - v59 )
      {
        sub_1984210((__int64 *)(a1 + 88), v68 + v59);
        v59 = *(unsigned int *)(a1 + 96);
      }
      v69 = *(_QWORD *)(a1 + 88) + 320 * v59;
      if ( v86 != v65 )
      {
        do
        {
          if ( v69 )
          {
            v70 = *(_QWORD *)v65;
            *(_DWORD *)(v69 + 16) = 0;
            *(_DWORD *)(v69 + 20) = 16;
            *(_QWORD *)v69 = v70;
            *(_QWORD *)(v69 + 8) = v69 + 24;
            if ( *(_DWORD *)(v65 + 16) )
              sub_1983D00(v69 + 8, v65 + 8, v69 + 24, v61, v62, v63);
            sub_16CCCB0((_QWORD *)(v69 + 152), v69 + 192, v65 + 152);
          }
          v65 += 320;
          v69 += 320;
        }
        while ( v86 != v65 );
        LODWORD(v59) = *(_DWORD *)(a1 + 96);
      }
      *(_DWORD *)(a1 + 96) = v68 + v59;
    }
LABEL_113:
    v50 = (__int64)v106;
    v51 = (unsigned __int64)&v106[320 * (unsigned int)v107];
  }
  else
  {
    v50 = (__int64)v106;
    v47 = 0;
    v51 = (unsigned __int64)&v106[320 * (unsigned int)v107];
  }
  if ( v50 != v51 )
  {
    do
    {
      v51 -= 320LL;
      v52 = *(_QWORD *)(v51 + 168);
      if ( v52 != *(_QWORD *)(v51 + 160) )
        _libc_free(v52);
      v53 = *(_QWORD *)(v51 + 8);
      if ( v53 != v51 + 24 )
        _libc_free(v53);
    }
    while ( v50 != v51 );
    v51 = (unsigned __int64)v106;
  }
  if ( (_BYTE *)v51 != v108 )
    _libc_free(v51);
LABEL_85:
  if ( v102 != v101 )
    _libc_free((unsigned __int64)v102);
  if ( v97 != v99 )
    _libc_free((unsigned __int64)v97);
LABEL_91:
  sub_1984040((__int64)v92);
  return v47;
}
