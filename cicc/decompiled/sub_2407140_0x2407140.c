// Function: sub_2407140
// Address: 0x2407140
//
__int64 __fastcall sub_2407140(__int64 a1, __int64 *a2, __int64 a3)
{
  __int64 v5; // r14
  __int64 *v6; // r12
  __int64 v7; // rax
  __int64 v8; // r14
  __int64 *v9; // r11
  unsigned int v10; // esi
  __int64 v11; // rdi
  unsigned int v12; // ecx
  _QWORD *v13; // rdx
  __int64 v14; // rax
  __int64 v15; // rax
  __int64 v16; // rbx
  __int64 v17; // rdx
  unsigned int v18; // r13d
  unsigned int v19; // edi
  __int64 *v20; // rcx
  __int64 v21; // rsi
  __int64 v22; // rcx
  __int64 v23; // rdx
  __int64 *v24; // rsi
  int v25; // eax
  unsigned int v26; // r13d
  unsigned int v27; // r8d
  __int64 *v28; // rcx
  __int64 v29; // rdi
  unsigned int v30; // esi
  __int64 v31; // rdi
  unsigned int v32; // ecx
  _QWORD *v33; // rdx
  __int64 v34; // rax
  int v35; // r10d
  _QWORD *v36; // r9
  int v37; // eax
  int v38; // eax
  __int64 v39; // r13
  __int64 *v40; // rbx
  __int64 result; // rax
  __int64 *j; // r12
  __int64 v43; // rsi
  int v44; // ecx
  int v45; // r9d
  int v46; // ecx
  int v47; // r9d
  __int64 v48; // rax
  __int64 v49; // rcx
  __int64 v50; // rsi
  unsigned int v51; // edx
  __int64 *v52; // rdi
  __int64 v53; // r8
  _QWORD *v54; // r10
  int v55; // eax
  int v56; // eax
  int v57; // edi
  __int64 v58; // rcx
  __int64 v59; // rsi
  __int64 *v60; // r8
  int v61; // edx
  unsigned int v62; // ecx
  __int64 *v63; // rdi
  __int64 v64; // r9
  int v65; // edx
  int v66; // edx
  __int64 v67; // rsi
  unsigned int v68; // r13d
  __int64 v69; // rcx
  int v70; // r9d
  _QWORD *v71; // rdi
  int v72; // edx
  int v73; // edx
  __int64 v74; // rsi
  unsigned int v75; // r13d
  __int64 v76; // rcx
  int v77; // r10d
  _QWORD *v78; // rdi
  int v79; // edx
  int v80; // edx
  __int64 v81; // rsi
  unsigned int v82; // r13d
  int v83; // r10d
  __int64 v84; // rcx
  int v85; // r8d
  unsigned int v86; // r10d
  int v87; // edx
  int v88; // edx
  __int64 v89; // rsi
  unsigned int v90; // r13d
  int v91; // r9d
  __int64 v92; // rcx
  int v93; // r8d
  unsigned int v94; // r9d
  int i; // edi
  int v96; // r10d
  int v97; // r10d
  int v98; // r8d
  unsigned int v99; // r10d
  int v100; // r8d
  unsigned int v101; // r9d
  int v103; // [rsp+10h] [rbp-A0h]
  __int64 *v104; // [rsp+10h] [rbp-A0h]
  __int64 *v105; // [rsp+10h] [rbp-A0h]
  __int64 *v106; // [rsp+10h] [rbp-A0h]
  __int64 *v107; // [rsp+10h] [rbp-A0h]
  __int64 v108; // [rsp+18h] [rbp-98h]
  __int64 v109; // [rsp+30h] [rbp-80h]
  __int64 v110; // [rsp+38h] [rbp-78h]
  __int64 v111; // [rsp+38h] [rbp-78h]
  __int64 v112; // [rsp+38h] [rbp-78h]
  __int64 v113; // [rsp+48h] [rbp-68h] BYREF
  _BYTE v114[96]; // [rsp+50h] [rbp-60h] BYREF

  v5 = *a2;
  v108 = a3 + 1752;
  v109 = *a2 + 96LL * *((unsigned int *)a2 + 2);
  if ( *a2 == v109 )
    goto LABEL_27;
  do
  {
    if ( *(_BYTE *)(v5 + 8) )
    {
      v48 = *(_QWORD *)v5;
      v49 = *(unsigned int *)(a1 + 96);
      v50 = *(_QWORD *)(a1 + 80);
      v113 = *(_QWORD *)v5;
      if ( !(_DWORD)v49 )
        goto LABEL_53;
      v51 = (v49 - 1) & (((unsigned int)v48 >> 9) ^ ((unsigned int)v48 >> 4));
      v52 = (__int64 *)(v50 + 8LL * v51);
      v53 = *v52;
      if ( v48 != *v52 )
      {
        v57 = 1;
        while ( v53 != -4096 )
        {
          v97 = v57 + 1;
          v51 = (v49 - 1) & (v57 + v51);
          v52 = (__int64 *)(v50 + 8LL * v51);
          v53 = *v52;
          if ( v48 == *v52 )
            goto LABEL_40;
          v57 = v97;
        }
LABEL_53:
        v58 = *(unsigned int *)(a1 + 128);
        v59 = *(_QWORD *)(a1 + 112);
        v60 = (__int64 *)(v59 + 8 * v58);
        if ( !(_DWORD)v58 )
          goto LABEL_115;
        v61 = v58 - 1;
        v62 = (v58 - 1) & (((unsigned int)v48 >> 9) ^ ((unsigned int)v48 >> 4));
        v63 = (__int64 *)(v59 + 8LL * v62);
        v64 = *v63;
        if ( v48 != *v63 )
        {
          for ( i = 1; ; i = v96 )
          {
            if ( v64 == -4096 )
              goto LABEL_115;
            v96 = i + 1;
            v62 = v61 & (i + v62);
            v63 = (__int64 *)(v59 + 8LL * v62);
            v64 = *v63;
            if ( v48 == *v63 )
              break;
          }
        }
        if ( v60 == v63 )
          goto LABEL_115;
        v112 = a1;
        sub_23FF920((__int64)v114, a3 + 904, &v113);
        a1 = v112;
        goto LABEL_3;
      }
LABEL_40:
      if ( v52 == (__int64 *)(v50 + 8 * v49) )
        goto LABEL_53;
      v111 = a1;
      sub_23FF920((__int64)v114, a3 + 872, &v113);
      a1 = v111;
    }
LABEL_3:
    v6 = *(__int64 **)(v5 + 16);
    v7 = *(unsigned int *)(v5 + 24);
    if ( &v6[v7] == v6 )
      goto LABEL_26;
    v110 = v5;
    v8 = a1;
    v9 = &v6[v7];
    do
    {
      while ( 1 )
      {
        v15 = *(unsigned int *)(v8 + 160);
        v16 = *v6;
        v17 = *(_QWORD *)(v8 + 144);
        if ( !(_DWORD)v15 )
          goto LABEL_11;
        v18 = ((unsigned int)v16 >> 9) ^ ((unsigned int)v16 >> 4);
        v19 = (v15 - 1) & v18;
        v20 = (__int64 *)(v17 + 8LL * v19);
        v21 = *v20;
        if ( v16 == *v20 )
        {
LABEL_10:
          if ( v20 == (__int64 *)(v17 + 8 * v15) )
            goto LABEL_11;
          v10 = *(_DWORD *)(a3 + 1744);
          if ( v10 )
          {
            v11 = *(_QWORD *)(a3 + 1728);
            v12 = (v10 - 1) & v18;
            v13 = (_QWORD *)(v11 + 8LL * v12);
            v14 = *v13;
            if ( v16 == *v13 )
              goto LABEL_7;
            v103 = 1;
            v54 = 0;
            while ( v14 != -4096 )
            {
              if ( v14 != -8192 || v54 )
                v13 = v54;
              v12 = (v10 - 1) & (v103 + v12);
              v14 = *(_QWORD *)(v11 + 8LL * v12);
              if ( v16 == v14 )
                goto LABEL_7;
              ++v103;
              v54 = v13;
              v13 = (_QWORD *)(v11 + 8LL * v12);
            }
            v55 = *(_DWORD *)(a3 + 1736);
            if ( !v54 )
              v54 = v13;
            ++*(_QWORD *)(a3 + 1720);
            v56 = v55 + 1;
            if ( 4 * v56 < 3 * v10 )
            {
              if ( v10 - *(_DWORD *)(a3 + 1740) - v56 <= v10 >> 3 )
              {
                v107 = v9;
                sub_2404320(a3 + 1720, v10);
                v87 = *(_DWORD *)(a3 + 1744);
                if ( !v87 )
                {
LABEL_117:
                  ++*(_DWORD *)(a3 + 1736);
                  BUG();
                }
                v88 = v87 - 1;
                v89 = *(_QWORD *)(a3 + 1728);
                v71 = 0;
                v9 = v107;
                v90 = v88 & v18;
                v91 = 1;
                v54 = (_QWORD *)(v89 + 8LL * v90);
                v92 = *v54;
                v56 = *(_DWORD *)(a3 + 1736) + 1;
                if ( v16 != *v54 )
                {
                  while ( v92 != -4096 )
                  {
                    if ( v92 == -8192 && !v71 )
                      v71 = v54;
                    v93 = v91 + 1;
                    v94 = v88 & (v90 + v91);
                    v54 = (_QWORD *)(v89 + 8LL * v94);
                    v90 = v94;
                    v92 = *v54;
                    if ( v16 == *v54 )
                      goto LABEL_48;
                    v91 = v93;
                  }
                  goto LABEL_62;
                }
              }
              goto LABEL_48;
            }
          }
          else
          {
            ++*(_QWORD *)(a3 + 1720);
          }
          v104 = v9;
          sub_2404320(a3 + 1720, 2 * v10);
          v65 = *(_DWORD *)(a3 + 1744);
          if ( !v65 )
            goto LABEL_117;
          v66 = v65 - 1;
          v67 = *(_QWORD *)(a3 + 1728);
          v9 = v104;
          v68 = v66 & v18;
          v54 = (_QWORD *)(v67 + 8LL * v68);
          v69 = *v54;
          v56 = *(_DWORD *)(a3 + 1736) + 1;
          if ( v16 != *v54 )
          {
            v70 = 1;
            v71 = 0;
            while ( v69 != -4096 )
            {
              if ( v69 == -8192 && !v71 )
                v71 = v54;
              v100 = v70 + 1;
              v101 = v66 & (v68 + v70);
              v54 = (_QWORD *)(v67 + 8LL * v101);
              v68 = v101;
              v69 = *v54;
              if ( v16 == *v54 )
                goto LABEL_48;
              v70 = v100;
            }
LABEL_62:
            if ( v71 )
              v54 = v71;
          }
LABEL_48:
          *(_DWORD *)(a3 + 1736) = v56;
          if ( *v54 != -4096 )
            --*(_DWORD *)(a3 + 1740);
          *v54 = v16;
          goto LABEL_7;
        }
        v44 = 1;
        while ( v21 != -4096 )
        {
          v45 = v44 + 1;
          v19 = (v15 - 1) & (v44 + v19);
          v20 = (__int64 *)(v17 + 8LL * v19);
          v21 = *v20;
          if ( v16 == *v20 )
            goto LABEL_10;
          v44 = v45;
        }
LABEL_11:
        v22 = *(unsigned int *)(v8 + 192);
        v23 = *(_QWORD *)(v8 + 176);
        v24 = (__int64 *)(v23 + 8 * v22);
        if ( !(_DWORD)v22 )
          goto LABEL_115;
        v25 = v22 - 1;
        v26 = ((unsigned int)v16 >> 9) ^ ((unsigned int)v16 >> 4);
        v27 = (v22 - 1) & v26;
        v28 = (__int64 *)(v23 + 8LL * v27);
        v29 = *v28;
        if ( v16 != *v28 )
        {
          v46 = 1;
          while ( v29 != -4096 )
          {
            v47 = v46 + 1;
            v27 = v25 & (v46 + v27);
            v28 = (__int64 *)(v23 + 8LL * v27);
            v29 = *v28;
            if ( v16 == *v28 )
              goto LABEL_13;
            v46 = v47;
          }
LABEL_115:
          BUG();
        }
LABEL_13:
        if ( v24 == v28 )
          goto LABEL_115;
        v30 = *(_DWORD *)(a3 + 1776);
        if ( !v30 )
        {
          ++*(_QWORD *)(a3 + 1752);
          goto LABEL_66;
        }
        v31 = *(_QWORD *)(a3 + 1760);
        v32 = (v30 - 1) & v26;
        v33 = (_QWORD *)(v31 + 8LL * v32);
        v34 = *v33;
        if ( v16 != *v33 )
          break;
LABEL_7:
        if ( v9 == ++v6 )
          goto LABEL_25;
      }
      v35 = 1;
      v36 = 0;
      while ( v34 != -4096 )
      {
        if ( v34 != -8192 || v36 )
          v33 = v36;
        v32 = (v30 - 1) & (v35 + v32);
        v34 = *(_QWORD *)(v31 + 8LL * v32);
        if ( v16 == v34 )
          goto LABEL_7;
        ++v35;
        v36 = v33;
        v33 = (_QWORD *)(v31 + 8LL * v32);
      }
      v37 = *(_DWORD *)(a3 + 1768);
      if ( !v36 )
        v36 = v33;
      ++*(_QWORD *)(a3 + 1752);
      v38 = v37 + 1;
      if ( 4 * v38 < 3 * v30 )
      {
        if ( v30 - *(_DWORD *)(a3 + 1772) - v38 <= v30 >> 3 )
        {
          v106 = v9;
          sub_2404320(v108, v30);
          v79 = *(_DWORD *)(a3 + 1776);
          if ( !v79 )
          {
LABEL_116:
            ++*(_DWORD *)(a3 + 1768);
            BUG();
          }
          v80 = v79 - 1;
          v81 = *(_QWORD *)(a3 + 1760);
          v78 = 0;
          v9 = v106;
          v82 = v80 & v26;
          v83 = 1;
          v36 = (_QWORD *)(v81 + 8LL * v82);
          v84 = *v36;
          v38 = *(_DWORD *)(a3 + 1768) + 1;
          if ( v16 != *v36 )
          {
            while ( v84 != -4096 )
            {
              if ( v84 == -8192 && !v78 )
                v78 = v36;
              v85 = v83 + 1;
              v86 = v80 & (v82 + v83);
              v36 = (_QWORD *)(v81 + 8LL * v86);
              v82 = v86;
              v84 = *v36;
              if ( v16 == *v36 )
                goto LABEL_22;
              v83 = v85;
            }
LABEL_70:
            if ( v78 )
              v36 = v78;
            goto LABEL_22;
          }
        }
        goto LABEL_22;
      }
LABEL_66:
      v105 = v9;
      sub_2404320(v108, 2 * v30);
      v72 = *(_DWORD *)(a3 + 1776);
      if ( !v72 )
        goto LABEL_116;
      v73 = v72 - 1;
      v74 = *(_QWORD *)(a3 + 1760);
      v9 = v105;
      v75 = v73 & v26;
      v36 = (_QWORD *)(v74 + 8LL * v75);
      v76 = *v36;
      v38 = *(_DWORD *)(a3 + 1768) + 1;
      if ( v16 != *v36 )
      {
        v77 = 1;
        v78 = 0;
        while ( v76 != -4096 )
        {
          if ( !v78 && v76 == -8192 )
            v78 = v36;
          v98 = v77 + 1;
          v99 = v73 & (v75 + v77);
          v36 = (_QWORD *)(v74 + 8LL * v99);
          v75 = v99;
          v76 = *v36;
          if ( v16 == *v36 )
            goto LABEL_22;
          v77 = v98;
        }
        goto LABEL_70;
      }
LABEL_22:
      *(_DWORD *)(a3 + 1768) = v38;
      if ( *v36 != -4096 )
        --*(_DWORD *)(a3 + 1772);
      ++v6;
      *v36 = v16;
    }
    while ( v9 != v6 );
LABEL_25:
    a1 = v8;
    v5 = v110;
LABEL_26:
    v5 += 96;
  }
  while ( v109 != v5 );
LABEL_27:
  v39 = a1;
  v40 = (__int64 *)a2[98];
  result = *((unsigned int *)a2 + 198);
  for ( j = &v40[result]; j != v40; result = sub_2407140(v39, v43, a3) )
    v43 = *v40++;
  return result;
}
