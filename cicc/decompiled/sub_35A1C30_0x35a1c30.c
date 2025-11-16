// Function: sub_35A1C30
// Address: 0x35a1c30
//
__int64 __fastcall sub_35A1C30(__int64 a1, unsigned int a2)
{
  __int64 v3; // r12
  __int64 *v4; // rax
  __int64 v5; // rax
  __int64 v6; // rdi
  int v7; // eax
  __int64 *v8; // r14
  unsigned int v9; // esi
  int v10; // r10d
  __int64 v11; // r8
  __int64 **v12; // rcx
  unsigned int v13; // edx
  __int64 **v14; // rax
  __int64 *v15; // rdi
  __int64 **v16; // rax
  __int64 *v17; // rdx
  unsigned int v18; // esi
  __int64 *v19; // r14
  __int64 v20; // r9
  int v21; // r11d
  __int64 **v22; // rdi
  unsigned int v23; // ecx
  __int64 **v24; // rax
  __int64 *v25; // r8
  __int64 **v26; // rax
  unsigned int v27; // esi
  __int64 *v28; // r14
  __int64 v29; // r9
  int v30; // r11d
  __int64 *v31; // rdx
  unsigned int j; // eax
  __int64 *v33; // rdi
  __int64 v34; // r8
  int v35; // eax
  int v36; // ecx
  int v37; // eax
  int v38; // edx
  __int64 *v39; // rdi
  __int64 **v41; // rdx
  __int64 *v42; // r14
  __int64 v43; // rcx
  unsigned int v44; // esi
  __int64 v45; // r9
  int v46; // r11d
  __int64 *v47; // rdx
  unsigned int v48; // eax
  __int64 *v49; // rdi
  __int64 v50; // r8
  __int64 **v51; // rdx
  int v52; // edi
  __int64 *v53; // rax
  __int64 v54; // rdi
  int v55; // ecx
  __int64 *v56; // rax
  __int64 v57; // r13
  __int64 *v58; // rax
  __int64 *v59; // rdx
  __int64 v60; // rax
  __int64 v61; // rdx
  __int64 v62; // r13
  __int64 *v63; // rdx
  __int64 v64; // rax
  __int64 v65; // rdx
  unsigned int v66; // eax
  unsigned int v67; // eax
  int v68; // eax
  int v69; // eax
  unsigned __int64 v70; // [rsp+8h] [rbp-68h]
  __int64 *i; // [rsp+18h] [rbp-58h] BYREF
  __int64 *v72; // [rsp+20h] [rbp-50h] BYREF
  __int64 *v73; // [rsp+28h] [rbp-48h] BYREF
  __int64 **v74; // [rsp+30h] [rbp-40h] BYREF
  __int64 *v75; // [rsp+38h] [rbp-38h]

  v3 = sub_37F36E0(a2, *(_QWORD *)(a1 + 48), *(_QWORD *)(a1 + 24), *(_QWORD *)(a1 + 32));
  if ( !a2 )
  {
    v4 = *(__int64 **)(a1 + 368);
    if ( v4 != (__int64 *)(*(_QWORD *)(a1 + 384) - 8LL) )
    {
      if ( v4 )
      {
        *v4 = v3;
        v4 = *(__int64 **)(a1 + 368);
      }
      *(_QWORD *)(a1 + 368) = v4 + 1;
      goto LABEL_8;
    }
    v57 = *(_QWORD *)(a1 + 392);
    if ( (((__int64)v4 - *(_QWORD *)(a1 + 376)) >> 3)
       + ((((v57 - *(_QWORD *)(a1 + 360)) >> 3) - 1) << 6)
       + ((__int64)(*(_QWORD *)(a1 + 352) - *(_QWORD *)(a1 + 336)) >> 3) != 0xFFFFFFFFFFFFFFFLL )
    {
      if ( (unsigned __int64)(*(_QWORD *)(a1 + 328) - ((v57 - *(_QWORD *)(a1 + 320)) >> 3)) <= 1 )
      {
        sub_359C1F0((unsigned __int64 *)(a1 + 320), 1u, 0);
        v57 = *(_QWORD *)(a1 + 392);
      }
      *(_QWORD *)(v57 + 8) = sub_22077B0(0x200u);
      v58 = *(__int64 **)(a1 + 368);
      if ( v58 )
        *v58 = v3;
      v59 = (__int64 *)(*(_QWORD *)(a1 + 392) + 8LL);
      *(_QWORD *)(a1 + 392) = v59;
      v60 = *v59;
      v61 = *v59 + 512;
      *(_QWORD *)(a1 + 376) = v60;
      *(_QWORD *)(a1 + 384) = v61;
      *(_QWORD *)(a1 + 368) = v60;
      goto LABEL_8;
    }
LABEL_115:
    sub_4262D8((__int64)"cannot create std::deque larger than max_size()");
  }
  v5 = *(_QWORD *)(a1 + 416);
  if ( v5 == *(_QWORD *)(a1 + 424) )
  {
    v62 = *(_QWORD *)(a1 + 440);
    if ( ((((*(_QWORD *)(a1 + 472) - v62) >> 3) - 1) << 6)
       + ((__int64)(*(_QWORD *)(a1 + 448) - *(_QWORD *)(a1 + 456)) >> 3)
       + ((*(_QWORD *)(a1 + 432) - v5) >> 3) != 0xFFFFFFFFFFFFFFFLL )
    {
      if ( v62 == *(_QWORD *)(a1 + 400) )
      {
        sub_359C1F0((unsigned __int64 *)(a1 + 400), 1u, 1);
        v62 = *(_QWORD *)(a1 + 440);
      }
      *(_QWORD *)(v62 - 8) = sub_22077B0(0x200u);
      v63 = (__int64 *)(*(_QWORD *)(a1 + 440) - 8LL);
      *(_QWORD *)(a1 + 440) = v63;
      v64 = *v63;
      v65 = *v63 + 512;
      *(_QWORD *)(a1 + 424) = v64;
      *(_QWORD *)(a1 + 432) = v65;
      *(_QWORD *)(a1 + 416) = v64 + 504;
      *(_QWORD *)(v64 + 504) = v3;
      goto LABEL_8;
    }
    goto LABEL_115;
  }
  *(_QWORD *)(v5 - 8) = v3;
  *(_QWORD *)(a1 + 416) -= 8LL;
LABEL_8:
  v6 = *(_QWORD *)(*(_QWORD *)(a1 + 48) + 56LL);
  v72 = *(__int64 **)(v3 + 56);
  for ( i = (__int64 *)v6; ; v6 = (__int64)i )
  {
    v7 = *(_DWORD *)(v6 + 44);
    if ( (v7 & 4) != 0 || (v7 & 8) == 0 )
      break;
    if ( sub_2E88A90(v6, 512, 1) )
      return v3;
LABEL_12:
    v8 = i;
    v9 = *(_DWORD *)(a1 + 280);
    v73 = i;
    if ( !v9 )
    {
      ++*(_QWORD *)(a1 + 256);
      v74 = 0;
LABEL_70:
      v9 *= 2;
LABEL_71:
      sub_2E48800(a1 + 256, v9);
      sub_3547B30(a1 + 256, (__int64 *)&v73, &v74);
      v39 = v73;
      v12 = v74;
      v38 = *(_DWORD *)(a1 + 272) + 1;
      goto LABEL_50;
    }
    v10 = 1;
    v11 = *(_QWORD *)(a1 + 264);
    v12 = 0;
    v13 = (v9 - 1) & (((unsigned int)i >> 9) ^ ((unsigned int)i >> 4));
    v14 = (__int64 **)(v11 + 16LL * v13);
    v15 = *v14;
    if ( i == *v14 )
    {
LABEL_14:
      v16 = v14 + 1;
      goto LABEL_15;
    }
    while ( v15 != (__int64 *)-4096LL )
    {
      if ( v15 == (__int64 *)-8192LL && !v12 )
        v12 = v14;
      v13 = (v9 - 1) & (v10 + v13);
      v14 = (__int64 **)(v11 + 16LL * v13);
      v15 = *v14;
      if ( i == *v14 )
        goto LABEL_14;
      ++v10;
    }
    if ( !v12 )
      v12 = v14;
    v37 = *(_DWORD *)(a1 + 272);
    ++*(_QWORD *)(a1 + 256);
    v38 = v37 + 1;
    v74 = v12;
    if ( 4 * (v37 + 1) >= 3 * v9 )
      goto LABEL_70;
    v39 = v8;
    if ( v9 - *(_DWORD *)(a1 + 276) - v38 <= v9 >> 3 )
      goto LABEL_71;
LABEL_50:
    *(_DWORD *)(a1 + 272) = v38;
    if ( *v12 != (__int64 *)-4096LL )
      --*(_DWORD *)(a1 + 276);
    *v12 = v39;
    v16 = v12 + 1;
    v12[1] = 0;
LABEL_15:
    *v16 = v8;
    v17 = v72;
    v18 = *(_DWORD *)(a1 + 280);
    v19 = i;
    v73 = v72;
    if ( !v18 )
    {
      ++*(_QWORD *)(a1 + 256);
      v74 = 0;
      goto LABEL_73;
    }
    v20 = *(_QWORD *)(a1 + 264);
    v21 = 1;
    v22 = 0;
    v23 = (v18 - 1) & (((unsigned int)v72 >> 9) ^ ((unsigned int)v72 >> 4));
    v24 = (__int64 **)(v20 + 16LL * v23);
    v25 = *v24;
    if ( v72 != *v24 )
    {
      while ( v25 != (__int64 *)-4096LL )
      {
        if ( !v22 && v25 == (__int64 *)-8192LL )
          v22 = v24;
        v23 = (v18 - 1) & (v21 + v23);
        v24 = (__int64 **)(v20 + 16LL * v23);
        v25 = *v24;
        if ( v72 == *v24 )
          goto LABEL_17;
        ++v21;
      }
      if ( !v22 )
        v22 = v24;
      v35 = *(_DWORD *)(a1 + 272);
      ++*(_QWORD *)(a1 + 256);
      v36 = v35 + 1;
      v74 = v22;
      if ( 4 * (v35 + 1) < 3 * v18 )
      {
        if ( v18 - *(_DWORD *)(a1 + 276) - v36 > v18 >> 3 )
        {
LABEL_37:
          *(_DWORD *)(a1 + 272) = v36;
          if ( *v22 != (__int64 *)-4096LL )
            --*(_DWORD *)(a1 + 276);
          *v22 = v17;
          v26 = v22 + 1;
          v22[1] = 0;
          goto LABEL_18;
        }
LABEL_74:
        sub_2E48800(a1 + 256, v18);
        sub_3547B30(a1 + 256, (__int64 *)&v73, &v74);
        v17 = v73;
        v22 = v74;
        v36 = *(_DWORD *)(a1 + 272) + 1;
        goto LABEL_37;
      }
LABEL_73:
      v18 *= 2;
      goto LABEL_74;
    }
LABEL_17:
    v26 = v24 + 1;
LABEL_18:
    *v26 = v19;
    v27 = *(_DWORD *)(a1 + 312);
    v28 = v72;
    v74 = (__int64 **)v3;
    v75 = i;
    if ( !v27 )
    {
      ++*(_QWORD *)(a1 + 288);
      v73 = 0;
LABEL_84:
      v27 *= 2;
      goto LABEL_85;
    }
    v29 = *(_QWORD *)(a1 + 296);
    v30 = 1;
    v31 = 0;
    v70 = (unsigned __int64)(((unsigned int)v3 >> 9) ^ ((unsigned int)v3 >> 4)) << 32;
    for ( j = (v27 - 1)
            & (((0xBF58476D1CE4E5B9LL * (v70 | ((unsigned int)i >> 9) ^ ((unsigned int)i >> 4))) >> 31)
             ^ (484763065 * (v70 | ((unsigned int)i >> 9) ^ ((unsigned int)i >> 4)))); ; j = (v27 - 1) & v66 )
    {
      v33 = (__int64 *)(v29 + 24LL * j);
      v34 = *v33;
      if ( v3 == *v33 && i == (__int64 *)v33[1] )
      {
        v41 = (__int64 **)(v33 + 2);
        goto LABEL_57;
      }
      if ( v34 == -4096 )
        break;
      if ( v34 == -8192 && v33[1] == -8192 && !v31 )
        v31 = (__int64 *)(v29 + 24LL * j);
LABEL_102:
      v66 = v30 + j;
      ++v30;
    }
    if ( v33[1] != -4096 )
      goto LABEL_102;
    v69 = *(_DWORD *)(a1 + 304);
    if ( !v31 )
      v31 = v33;
    ++*(_QWORD *)(a1 + 288);
    v55 = v69 + 1;
    v73 = v31;
    if ( 4 * (v69 + 1) >= 3 * v27 )
      goto LABEL_84;
    v54 = v3;
    if ( v27 - *(_DWORD *)(a1 + 308) - v55 <= v27 >> 3 )
    {
LABEL_85:
      sub_35A1120(a1 + 288, v27);
      sub_359BDE0(a1 + 288, (__int64 *)&v74, &v73);
      v54 = (__int64)v74;
      v31 = v73;
      v55 = *(_DWORD *)(a1 + 304) + 1;
    }
    *(_DWORD *)(a1 + 304) = v55;
    if ( *v31 != -4096 || v31[1] != -4096 )
      --*(_DWORD *)(a1 + 308);
    *v31 = v54;
    v56 = v75;
    v41 = (__int64 **)(v31 + 2);
    *v41 = 0;
    *(v41 - 1) = v56;
LABEL_57:
    *v41 = v28;
    v42 = i;
    v43 = *(_QWORD *)(a1 + 48);
    v44 = *(_DWORD *)(a1 + 312);
    v75 = i;
    v74 = (__int64 **)v43;
    if ( !v44 )
    {
      ++*(_QWORD *)(a1 + 288);
      v73 = 0;
LABEL_76:
      v44 *= 2;
      goto LABEL_77;
    }
    v45 = *(_QWORD *)(a1 + 296);
    v46 = 1;
    v47 = 0;
    v48 = (v44 - 1)
        & (((0xBF58476D1CE4E5B9LL
           * (((unsigned int)i >> 9) ^ ((unsigned int)i >> 4)
            | ((unsigned __int64)(((unsigned int)v43 >> 9) ^ ((unsigned int)v43 >> 4)) << 32))) >> 31)
         ^ (484763065 * (((unsigned int)i >> 9) ^ ((unsigned int)i >> 4))));
    while ( 2 )
    {
      v49 = (__int64 *)(v45 + 24LL * v48);
      v50 = *v49;
      if ( v43 == *v49 && i == (__int64 *)v49[1] )
      {
        v51 = (__int64 **)(v49 + 2);
        goto LABEL_68;
      }
      if ( v50 != -4096 )
      {
        if ( v50 == -8192 && v49[1] == -8192 && !v47 )
          v47 = (__int64 *)(v45 + 24LL * v48);
        goto LABEL_104;
      }
      if ( v49[1] != -4096 )
      {
LABEL_104:
        v67 = v46 + v48;
        ++v46;
        v48 = (v44 - 1) & v67;
        continue;
      }
      break;
    }
    v68 = *(_DWORD *)(a1 + 304);
    if ( !v47 )
      v47 = v49;
    ++*(_QWORD *)(a1 + 288);
    v52 = v68 + 1;
    v73 = v47;
    if ( 4 * (v68 + 1) >= 3 * v44 )
      goto LABEL_76;
    if ( v44 - *(_DWORD *)(a1 + 308) - v52 <= v44 >> 3 )
    {
LABEL_77:
      sub_35A1120(a1 + 288, v44);
      sub_359BDE0(a1 + 288, (__int64 *)&v74, &v73);
      v43 = (__int64)v74;
      v47 = v73;
      v52 = *(_DWORD *)(a1 + 304) + 1;
    }
    *(_DWORD *)(a1 + 304) = v52;
    if ( *v47 != -4096 || v47[1] != -4096 )
      --*(_DWORD *)(a1 + 308);
    *v47 = v43;
    v53 = v75;
    v51 = (__int64 **)(v47 + 2);
    *v51 = 0;
    *(v51 - 1) = v53;
LABEL_68:
    *v51 = v42;
    sub_2FD79B0((__int64 *)&i);
    sub_2FD79B0((__int64 *)&v72);
  }
  if ( (*(_QWORD *)(*(_QWORD *)(v6 + 16) + 24LL) & 0x200LL) == 0 )
    goto LABEL_12;
  return v3;
}
