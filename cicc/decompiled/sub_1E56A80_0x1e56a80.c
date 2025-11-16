// Function: sub_1E56A80
// Address: 0x1e56a80
//
__int64 __fastcall sub_1E56A80(__int64 a1, __int64 a2, unsigned __int64 a3, __int64 *a4)
{
  __int64 v5; // rax
  int v6; // eax
  unsigned __int64 v7; // r11
  unsigned __int64 *v8; // r12
  __int64 v10; // rax
  __int64 v11; // rbx
  unsigned __int64 v12; // rsi
  int v13; // eax
  char v14; // r10
  __int64 v15; // r15
  __int64 (*v16)(); // rax
  __int64 v17; // rax
  __int64 v18; // r8
  int v19; // r9d
  __int64 v20; // rdi
  __int64 (*v21)(); // rax
  char v22; // r11
  char v23; // r11
  unsigned int v24; // eax
  _QWORD *v25; // r10
  _QWORD *v26; // rbx
  unsigned __int64 v27; // rsi
  unsigned int v28; // eax
  _QWORD *v29; // r10
  _QWORD *v30; // rbx
  unsigned __int64 v31; // rsi
  char v32; // r14
  unsigned int v33; // r15d
  __int64 v34; // r10
  _QWORD *v35; // r8
  _QWORD *v36; // r11
  __int64 v37; // r14
  __int64 v38; // rcx
  __int64 v39; // rdx
  __int64 *v40; // rax
  _QWORD *v41; // r11
  __int64 v42; // rax
  __int64 v43; // rdx
  __int64 *v44; // rax
  __int64 v45; // rcx
  _QWORD *v46; // r11
  __int64 v47; // rax
  __int64 result; // rax
  char v49; // al
  __int64 v50; // rax
  __int64 v51; // rsi
  unsigned int v52; // r8d
  __int64 *v53; // rdx
  __int64 v54; // rdi
  __int64 v55; // rcx
  __int64 v56; // rax
  __int64 v57; // rdx
  int v58; // eax
  int v59; // esi
  char v60; // r8
  int v61; // edx
  __int64 v62; // rax
  __int64 v63; // rdx
  unsigned int v64; // eax
  _QWORD *v65; // rcx
  __int64 v66; // r8
  _QWORD *v67; // rdx
  _QWORD *v68; // rax
  __int64 v69; // r9
  __int64 v70; // rdi
  int v71; // r9d
  _QWORD *v72; // r11
  unsigned __int64 v73; // rdi
  int v74; // r15d
  __int64 v75; // rdx
  __int64 v76; // rax
  unsigned __int64 *v77; // rax
  _QWORD *v78; // rax
  _QWORD *v79; // rcx
  __int64 v80; // rcx
  __int64 v81; // rdx
  __int64 *v82; // rax
  _QWORD *v83; // r11
  __int64 v84; // rax
  __int64 v85; // rdx
  __int64 *v86; // rax
  __int64 v87; // rcx
  int v88; // edx
  int v89; // r11d
  __int64 v90; // rdx
  __int64 v91; // [rsp+0h] [rbp-100h]
  _QWORD *v93; // [rsp+8h] [rbp-F8h]
  __int64 v95; // [rsp+18h] [rbp-E8h]
  unsigned __int64 *v96; // [rsp+28h] [rbp-D8h]
  __int64 v97; // [rsp+30h] [rbp-D0h]
  unsigned __int64 *v98; // [rsp+38h] [rbp-C8h]
  char v99; // [rsp+45h] [rbp-BBh]
  char v100; // [rsp+46h] [rbp-BAh]
  char v101; // [rsp+47h] [rbp-B9h]
  char v102[4]; // [rsp+48h] [rbp-B8h]
  char v103[4]; // [rsp+4Ch] [rbp-B4h]
  unsigned int v104; // [rsp+50h] [rbp-B0h]
  int v105; // [rsp+54h] [rbp-ACh]
  int v106; // [rsp+58h] [rbp-A8h]
  __int64 v107; // [rsp+58h] [rbp-A8h]
  __int64 v108; // [rsp+58h] [rbp-A8h]
  int v109; // [rsp+58h] [rbp-A8h]
  __int64 v110; // [rsp+60h] [rbp-A0h]
  __int64 v111; // [rsp+60h] [rbp-A0h]
  unsigned __int64 v112; // [rsp+68h] [rbp-98h] BYREF
  __int64 *v113; // [rsp+70h] [rbp-90h] BYREF
  __int64 v114; // [rsp+78h] [rbp-88h]
  __int64 v115; // [rsp+80h] [rbp-80h]
  __int64 *v116; // [rsp+88h] [rbp-78h]
  __int64 v117[4]; // [rsp+90h] [rbp-70h] BYREF
  __int64 *v118; // [rsp+B0h] [rbp-50h] BYREF
  __int64 v119; // [rsp+B8h] [rbp-48h]
  __int64 v120; // [rsp+C0h] [rbp-40h]
  __int64 *v121; // [rsp+C8h] [rbp-38h]

  v5 = *(_QWORD *)(a3 + 8);
  v112 = a3;
  v97 = v5;
  v6 = sub_1E404B0(a1, a3);
  v8 = (unsigned __int64 *)a4[2];
  v105 = v6;
  v98 = (unsigned __int64 *)a4[4];
  v96 = (unsigned __int64 *)a4[6];
  v95 = a4[5];
  if ( v8 == v96 )
    goto LABEL_107;
  v104 = 0;
  *(_DWORD *)v103 = 0;
  *(_DWORD *)v102 = 0;
  v99 = 0;
  v101 = 0;
  v100 = 0;
  do
  {
    v10 = *(unsigned int *)(v97 + 40);
    if ( !(_DWORD)v10 )
      goto LABEL_23;
    v11 = 0;
    v110 = 40 * v10;
    do
    {
      while ( 1 )
      {
        v15 = v11 + *(_QWORD *)(v97 + 32);
        if ( *(_BYTE *)v15 )
          goto LABEL_11;
        v106 = *(_DWORD *)(v15 + 8);
        if ( v106 >= 0 )
          goto LABEL_11;
        v16 = *(__int64 (**)())(**(_QWORD **)(a1 + 144) + 40LL);
        if ( v16 == sub_1D00B00 )
          BUG();
        v17 = v16();
        v19 = v106;
        v20 = v17;
        v21 = *(__int64 (**)())(*(_QWORD *)v17 + 600LL);
        if ( v21 != sub_1E40450 )
        {
          v49 = ((__int64 (__fastcall *)(__int64, __int64, __int64 *, __int64 **, __int64, _QWORD))v21)(
                  v20,
                  v97,
                  v117,
                  &v118,
                  v18,
                  (unsigned int)v106);
          v19 = v106;
          if ( v49 )
          {
            if ( *(_DWORD *)(*(_QWORD *)(v97 + 32) + 40LL * LODWORD(v117[0]) + 8) == v106 )
            {
              v50 = *(unsigned int *)(a2 + 2336);
              if ( (_DWORD)v50 )
              {
                v51 = *(_QWORD *)(a2 + 2320);
                v52 = (v50 - 1) & (((unsigned int)v112 >> 9) ^ ((unsigned int)v112 >> 4));
                v53 = (__int64 *)(v51 + 24LL * v52);
                v54 = *v53;
                if ( v112 == *v53 )
                {
LABEL_57:
                  if ( v53 != (__int64 *)(v51 + 24 * v50) && *((_DWORD *)v53 + 2) )
                    v19 = *((_DWORD *)v53 + 2);
                }
                else
                {
                  v88 = 1;
                  while ( v54 != -8 )
                  {
                    v89 = v88 + 1;
                    v90 = ((_DWORD)v50 - 1) & (v52 + v88);
                    v52 = v90;
                    v53 = (__int64 *)(v51 + 24 * v90);
                    v54 = *v53;
                    if ( v112 == *v53 )
                      goto LABEL_57;
                    v88 = v89;
                  }
                }
              }
            }
          }
        }
        v22 = sub_1E166B0(*(_QWORD *)(*v8 + 8), v19, 0);
        if ( (*(_BYTE *)(v15 + 3) & 0x10) != 0 )
          break;
        v12 = *v8;
        v13 = sub_1E404B0(a1, *v8);
        if ( !v14 )
        {
          if ( v105 == v13 && !*(_BYTE *)v15 )
          {
            v55 = *(_QWORD *)(v12 + 8);
            v107 = v55;
            if ( **(_WORD **)(v55 + 16) )
            {
              if ( **(_WORD **)(v55 + 16) != 45 )
              {
                v56 = sub_1E69D00(*(_QWORD *)(a1 + 152), *(unsigned int *)(v15 + 8));
                v57 = v56;
                if ( v56 )
                {
                  v58 = **(unsigned __int16 **)(v56 + 16);
                  if ( (!v58 || v58 == 45) && *(_QWORD *)(v57 + 24) == *(_QWORD *)(v107 + 24) )
                  {
                    v91 = v107;
                    v108 = v57;
                    if ( (unsigned __int8)sub_1E45F30(a1, a2, v57) )
                    {
                      v59 = sub_1E40FE0(*(_QWORD *)(v108 + 32), *(_DWORD *)(v108 + 40), *(_QWORD *)(v108 + 24));
                      v61 = *(_DWORD *)(v91 + 40);
                      if ( v61 )
                      {
                        v62 = *(_QWORD *)(v91 + 32);
                        v63 = v62 + 40LL * (unsigned int)(v61 - 1) + 40;
                        while ( *(_BYTE *)v62 || (*(_BYTE *)(v62 + 3) & 0x10) == 0 || v59 != *(_DWORD *)(v62 + 8) )
                        {
                          v62 += 40;
                          if ( v63 == v62 )
                            goto LABEL_11;
                        }
                        if ( !*(_DWORD *)v103 )
                        {
                          v99 = v60;
                          *(_DWORD *)v103 = v104;
                        }
                      }
                    }
                  }
                }
              }
            }
          }
          goto LABEL_11;
        }
        if ( v105 == v13 )
        {
          v65 = *(_QWORD **)(a1 + 48);
          v66 = a1 + 40;
          if ( v65 )
          {
            v67 = (_QWORD *)(a1 + 40);
            v68 = *(_QWORD **)(a1 + 48);
            do
            {
              while ( 1 )
              {
                v69 = v68[2];
                v70 = v68[3];
                if ( v68[4] >= v12 )
                  break;
                v68 = (_QWORD *)v68[3];
                if ( !v70 )
                  goto LABEL_87;
              }
              v67 = v68;
              v68 = (_QWORD *)v68[2];
            }
            while ( v69 );
LABEL_87:
            if ( v67 != (_QWORD *)v66 && v67[4] > v12 )
              v67 = (_QWORD *)(a1 + 40);
            v71 = *(_DWORD *)(a1 + 128);
            v72 = (_QWORD *)(a1 + 40);
            v109 = *(_DWORD *)(a1 + 136);
            v73 = v112;
            v74 = (*((_DWORD *)v67 + 10) - v71) % v109;
            do
            {
              while ( 1 )
              {
                v75 = v65[2];
                v76 = v65[3];
                if ( v65[4] >= v112 )
                  break;
                v65 = (_QWORD *)v65[3];
                if ( !v76 )
                  goto LABEL_94;
              }
              v72 = v65;
              v65 = (_QWORD *)v65[2];
            }
            while ( v75 );
LABEL_94:
            if ( (_QWORD *)v66 != v72 && v72[4] <= v112 )
              v66 = (__int64)v72;
          }
          else
          {
            v71 = *(_DWORD *)(a1 + 128);
            v73 = v112;
            v109 = *(_DWORD *)(a1 + 136);
            v74 = (*(_DWORD *)(a1 + 80) - v71) % v109;
          }
          if ( (*(_DWORD *)(v66 + 40) - v71) % v109 == v74 )
          {
            v78 = *(_QWORD **)(v12 + 112);
            v79 = &v78[2 * *(unsigned int *)(v12 + 120)];
            if ( v78 == v79 )
            {
LABEL_78:
              v64 = *(_DWORD *)v103;
              v100 = v14;
              if ( !*(_DWORD *)v103 )
                v64 = v104;
              *(_DWORD *)v103 = v64;
              goto LABEL_11;
            }
            while ( v73 != (*v78 & 0xFFFFFFFFFFFFFFF8LL) )
            {
              v78 += 2;
              if ( v79 == v78 )
                goto LABEL_78;
            }
          }
          v101 = v14;
          *(_DWORD *)v102 = v104;
        }
        else if ( v105 >= v13 )
        {
          if ( v105 > v13 )
            goto LABEL_78;
        }
        else
        {
          if ( *(_DWORD *)v103 )
            goto LABEL_100;
          if ( v104 )
          {
            *(_DWORD *)v103 = v104;
LABEL_100:
            v101 = v14;
            v100 = v14;
            *(_DWORD *)v102 = v104 - 1;
            goto LABEL_11;
          }
          v100 = v14;
        }
LABEL_11:
        v11 += 40;
        if ( v110 == v11 )
          goto LABEL_22;
      }
      if ( !v22 )
        goto LABEL_11;
      if ( (int)sub_1E404B0(a1, *v8) > v105 )
      {
        v101 = v23;
        *(_DWORD *)v102 = v104;
        goto LABEL_11;
      }
      v24 = *(_DWORD *)v103;
      v100 = v23;
      if ( !*(_DWORD *)v103 )
        v24 = v104;
      v11 += 40;
      *(_DWORD *)v103 = v24;
    }
    while ( v110 != v11 );
LABEL_22:
    v7 = v112;
LABEL_23:
    v25 = *(_QWORD **)(v7 + 112);
    v26 = &v25[2 * *(unsigned int *)(v7 + 120)];
    if ( v25 != v26 )
    {
      v27 = *v8;
      do
      {
        while ( v27 != (*v25 & 0xFFFFFFFFFFFFFFF8LL)
             || (((unsigned __int8)*v25 ^ 6) & 6) != 0
             || (unsigned int)sub_1E404B0(a1, v27) != v105 )
        {
          v25 += 2;
          if ( v26 == v25 )
            goto LABEL_32;
        }
        v28 = *(_DWORD *)v103;
        v100 = 1;
        if ( *(_DWORD *)v103 > v104 )
          v28 = v104;
        v25 += 2;
        *(_DWORD *)v103 = v28;
      }
      while ( v26 != v25 );
    }
LABEL_32:
    v29 = *(_QWORD **)(v7 + 32);
    v30 = &v29[2 * *(unsigned int *)(v7 + 40)];
    if ( v29 != v30 )
    {
      v31 = *v8;
      v32 = v101;
      v33 = *(_DWORD *)v102;
      do
      {
        while ( v31 != (*v29 & 0xFFFFFFFFFFFFFFF8LL) || (((unsigned __int8)*v29 ^ 6) & 6) != 0 )
        {
          v29 += 2;
          if ( v30 == v29 )
            goto LABEL_40;
        }
        if ( (unsigned int)sub_1E404B0(a1, v31) == v105 )
        {
          v33 = v104;
          v32 = 1;
        }
        v29 = (_QWORD *)(v34 + 16);
      }
      while ( v30 != v29 );
LABEL_40:
      v101 = v32;
      *(_DWORD *)v102 = v33;
    }
    if ( v98 == ++v8 )
    {
      v8 = *(unsigned __int64 **)(v95 + 8);
      v95 += 8;
      v98 = v8 + 64;
    }
    ++v104;
  }
  while ( v8 != v96 );
  if ( !v100 || !v101 )
  {
    if ( v99 )
      goto LABEL_102;
    goto LABEL_104;
  }
  if ( *(_DWORD *)v102 == *(_DWORD *)v103 )
    goto LABEL_107;
  if ( !v99 )
    goto LABEL_48;
LABEL_102:
  v100 = (*(_DWORD *)v102 < *(_DWORD *)v103) | v101 ^ 1;
  if ( *(_DWORD *)v102 < *(_DWORD *)v103 && v101 )
  {
LABEL_48:
    if ( *(unsigned int *)v103 >= (unsigned __int64)sub_1E47DE0(a4 + 6, a4 + 2) )
      sub_222CF80("deque::_M_range_check: __n (which is %zu)>= this->size() (which is %zu)", v103[0]);
    v118 = (__int64 *)a4[2];
    v119 = a4[3];
    v120 = a4[4];
    v121 = (__int64 *)a4[5];
    sub_1E40F70((__int64 *)&v118, *(unsigned int *)v103);
    v111 = *v118;
    if ( *(unsigned int *)v102 >= (unsigned __int64)sub_1E47DE0(a4 + 6, v35) )
      sub_222CF80("deque::_M_range_check: __n (which is %zu)>= this->size() (which is %zu)", v102[0]);
    v118 = (__int64 *)a4[2];
    v119 = a4[3];
    v120 = a4[4];
    v121 = (__int64 *)a4[5];
    sub_1E40F70((__int64 *)&v118, *(unsigned int *)v102);
    v37 = *v118;
    if ( *(_DWORD *)v102 >= *(_DWORD *)v103 )
    {
      v80 = a4[3];
      v81 = a4[4];
      v82 = (__int64 *)a4[5];
      v113 = (__int64 *)a4[2];
      v114 = v80;
      v115 = v81;
      v116 = v82;
      sub_1E40F70((__int64 *)&v113, *(unsigned int *)v102);
      v93 = v83;
      v118 = v113;
      v84 = *v116;
      v121 = v116;
      v119 = v84;
      v120 = v84 + 512;
      sub_1E56730(v117, v83, (__int64 *)&v118);
      v85 = v93[4];
      v86 = (__int64 *)v93[5];
      v87 = v93[3];
      v113 = (__int64 *)v93[2];
      v115 = v85;
      v116 = v86;
      v114 = v87;
      sub_1E40F70((__int64 *)&v113, *(unsigned int *)v103);
    }
    else
    {
      v38 = v36[3];
      v39 = v36[4];
      v40 = (__int64 *)v36[5];
      v113 = (__int64 *)v36[2];
      v114 = v38;
      v115 = v39;
      v116 = v40;
      sub_1E40F70((__int64 *)&v113, *(unsigned int *)v103);
      v93 = v41;
      v118 = v113;
      v42 = *v116;
      v121 = v116;
      v119 = v42;
      v120 = v42 + 512;
      sub_1E56730(v117, v41, (__int64 *)&v118);
      v43 = v93[4];
      v44 = (__int64 *)v93[5];
      v45 = v93[3];
      v113 = (__int64 *)v93[2];
      v115 = v43;
      v116 = v44;
      v114 = v45;
      sub_1E40F70((__int64 *)&v113, *(unsigned int *)v102);
    }
    v118 = v113;
    v47 = *v116;
    v121 = v116;
    v119 = v47;
    v120 = v47 + 512;
    sub_1E56730(v117, v46, (__int64 *)&v118);
    sub_1E56A80(a1, a2, v111, v93);
    sub_1E56A80(a1, a2, v112, v93);
    return sub_1E56A80(a1, a2, v37, v93);
  }
  else
  {
LABEL_104:
    if ( v100 )
    {
      result = a4[2];
      if ( result == a4[3] )
        return sub_1E48DF0(a4, &v112);
      *(_QWORD *)(result - 8) = v7;
      a4[2] -= 8;
      return result;
    }
LABEL_107:
    v77 = (unsigned __int64 *)a4[6];
    if ( v77 == (unsigned __int64 *)(a4[8] - 8) )
    {
      return sub_1E48ED0(a4, &v112);
    }
    else
    {
      if ( v77 )
      {
        *v77 = v7;
        v77 = (unsigned __int64 *)a4[6];
      }
      result = (__int64)(v77 + 1);
      a4[6] = result;
    }
  }
  return result;
}
