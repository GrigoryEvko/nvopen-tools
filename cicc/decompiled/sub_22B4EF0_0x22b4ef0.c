// Function: sub_22B4EF0
// Address: 0x22b4ef0
//
__int64 __fastcall sub_22B4EF0(__int64 a1, unsigned int a2, int a3, __int64 a4, __int64 a5)
{
  __int64 v6; // r11
  unsigned int v7; // esi
  __int64 v8; // r9
  __int64 *v9; // rbx
  __int64 *v10; // r15
  int v11; // r10d
  _QWORD *v12; // rdx
  unsigned int v13; // edi
  _QWORD *v14; // rax
  __int64 v15; // rcx
  __int64 v16; // r13
  int v17; // eax
  int v18; // edi
  __int64 v19; // rsi
  unsigned int v20; // ecx
  int v21; // eax
  __int64 v22; // r8
  int v23; // eax
  unsigned int v24; // esi
  __int64 v25; // r9
  unsigned int v26; // edi
  int *v27; // rcx
  int v28; // edx
  __int64 v29; // rax
  int v30; // r13d
  __int64 *v31; // r8
  unsigned int v32; // ecx
  __int64 *v33; // rdx
  __int64 v34; // rdi
  int v35; // eax
  int v36; // eax
  int v37; // ecx
  __int64 v38; // rdi
  _QWORD *v39; // r8
  unsigned int v40; // r14d
  int v41; // r9d
  __int64 v42; // rsi
  int v43; // edx
  int v44; // edx
  int v45; // edx
  int v46; // esi
  int v47; // esi
  __int64 v48; // r8
  unsigned int v49; // edi
  int *v50; // rcx
  int v51; // eax
  __int64 v52; // rdi
  __int64 v53; // rsi
  int v55; // r14d
  int *v56; // r8
  int v57; // edx
  int v58; // ecx
  int *v59; // r9
  _DWORD *v60; // rax
  _DWORD *v61; // rdx
  __int64 *v62; // r13
  __int64 *v63; // rbx
  __int64 v64; // r14
  unsigned int v65; // esi
  __int64 v66; // r8
  int v67; // r10d
  __int64 v68; // rdi
  unsigned int v69; // edx
  __int64 *v70; // rax
  __int64 v71; // rcx
  int v72; // eax
  int v73; // edx
  __int64 v74; // rcx
  int v75; // eax
  int v76; // ecx
  int v77; // ecx
  __int64 v78; // r8
  unsigned int v79; // esi
  _DWORD *v80; // rdx
  int v81; // edi
  _DWORD *v82; // r9
  _DWORD *v83; // rdx
  int v84; // ebx
  int v85; // r10d
  int v86; // r10d
  _QWORD *v87; // r9
  __int64 v90; // [rsp+20h] [rbp-90h]
  __int64 v91; // [rsp+20h] [rbp-90h]
  __int64 v92; // [rsp+20h] [rbp-90h]
  __int64 v93; // [rsp+20h] [rbp-90h]
  __int64 v94; // [rsp+20h] [rbp-90h]
  __int64 v95; // [rsp+28h] [rbp-88h]
  __int64 v96; // [rsp+30h] [rbp-80h]
  int v97; // [rsp+38h] [rbp-78h]
  unsigned int v98; // [rsp+3Ch] [rbp-74h]
  int v99; // [rsp+4Ch] [rbp-64h] BYREF
  __int64 v100; // [rsp+50h] [rbp-60h] BYREF
  __int64 v101; // [rsp+58h] [rbp-58h] BYREF
  __int64 *v102; // [rsp+60h] [rbp-50h] BYREF
  __int64 *v103; // [rsp+68h] [rbp-48h]
  __int64 v104; // [rsp+70h] [rbp-40h]
  __int64 v105; // [rsp+78h] [rbp-38h]

  v98 = a2;
  *(_DWORD *)a1 = a2;
  *(_DWORD *)(a1 + 4) = a3;
  v96 = a1 + 24;
  *(_QWORD *)(a1 + 8) = 0;
  *(_QWORD *)(a1 + 16) = 0;
  *(_QWORD *)(a1 + 24) = 0;
  *(_QWORD *)(a1 + 32) = 0;
  *(_QWORD *)(a1 + 40) = 0;
  *(_DWORD *)(a1 + 48) = 0;
  v95 = a1 + 56;
  *(_QWORD *)(a1 + 56) = 0;
  *(_QWORD *)(a1 + 64) = 0;
  *(_QWORD *)(a1 + 72) = 0;
  *(_DWORD *)(a1 + 80) = 0;
  *(_QWORD *)(a1 + 88) = 0;
  *(_QWORD *)(a1 + 96) = 0;
  *(_QWORD *)(a1 + 104) = 0;
  *(_DWORD *)(a1 + 112) = 0;
  *(_QWORD *)(a1 + 120) = 0;
  *(_QWORD *)(a1 + 128) = 0;
  *(_QWORD *)(a1 + 136) = 0;
  *(_DWORD *)(a1 + 144) = 0;
  v99 = 1;
  v97 = a2 + a3;
  if ( a2 < a2 + a3 )
  {
    v6 = a4;
    v7 = 0;
    v8 = 0;
    while ( 1 )
    {
      v9 = *(__int64 **)(v6 + 24);
      v10 = &v9[*(unsigned int *)(v6 + 32)];
      if ( v9 != v10 )
        break;
LABEL_16:
      v29 = *(_QWORD *)(v6 + 16);
      v101 = v29;
      if ( !v7 )
      {
        ++*(_QWORD *)(a1 + 24);
        v102 = 0;
LABEL_101:
        v94 = v6;
        v7 *= 2;
        goto LABEL_102;
      }
      v30 = 1;
      v31 = 0;
      v32 = (v7 - 1) & (((unsigned int)v29 >> 9) ^ ((unsigned int)v29 >> 4));
      v33 = (__int64 *)(v8 + 16LL * v32);
      v34 = *v33;
      if ( v29 == *v33 )
      {
LABEL_18:
        ++v98;
        v6 = *(_QWORD *)(v6 + 8);
        if ( v98 == v97 )
          goto LABEL_49;
        goto LABEL_19;
      }
      while ( v34 != -4096 )
      {
        if ( v31 || v34 != -8192 )
          v33 = v31;
        v32 = (v7 - 1) & (v30 + v32);
        v34 = *(_QWORD *)(v8 + 16LL * v32);
        if ( v29 == v34 )
          goto LABEL_18;
        ++v30;
        v31 = v33;
        v33 = (__int64 *)(v8 + 16LL * v32);
      }
      if ( !v31 )
        v31 = v33;
      v43 = *(_DWORD *)(a1 + 40);
      ++*(_QWORD *)(a1 + 24);
      v44 = v43 + 1;
      v102 = v31;
      if ( 4 * v44 >= 3 * v7 )
        goto LABEL_101;
      if ( v7 - (v44 + *(_DWORD *)(a1 + 44)) > v7 >> 3 )
        goto LABEL_44;
      v94 = v6;
LABEL_102:
      sub_D39D40(v96, v7);
      sub_22B1A50(v96, &v101, &v102);
      v29 = v101;
      v31 = v102;
      v6 = v94;
      v44 = *(_DWORD *)(a1 + 40) + 1;
LABEL_44:
      *(_DWORD *)(a1 + 40) = v44;
      if ( *v31 != -4096 )
        --*(_DWORD *)(a1 + 44);
      *v31 = v29;
      v45 = v99;
      *((_DWORD *)v31 + 2) = v99;
      v46 = *(_DWORD *)(a1 + 80);
      if ( v46 )
      {
        v47 = v46 - 1;
        v48 = *(_QWORD *)(a1 + 64);
        v49 = v47 & (37 * v45);
        v50 = (int *)(v48 + 16LL * v49);
        v51 = *v50;
        if ( v45 == *v50 )
          goto LABEL_48;
        v84 = 1;
        v59 = 0;
        while ( v51 != -1 )
        {
          if ( v51 != -2 || v59 )
            v50 = v59;
          v49 = v47 & (v84 + v49);
          v51 = *(_DWORD *)(v48 + 16LL * v49);
          if ( v45 == v51 )
            goto LABEL_48;
          ++v84;
          v59 = v50;
          v50 = (int *)(v48 + 16LL * v49);
        }
        if ( !v59 )
          v59 = v50;
      }
      else
      {
        v59 = 0;
      }
      v93 = v6;
      v60 = sub_22B3610(v95, &v99, v59);
      v6 = v93;
      v61 = v60;
      v51 = v99;
      *v61 = v99;
      *((_QWORD *)v61 + 1) = *(_QWORD *)(v93 + 16);
LABEL_48:
      ++v98;
      v6 = *(_QWORD *)(v6 + 8);
      v99 = v51 + 1;
      if ( v98 == v97 )
        goto LABEL_49;
LABEL_19:
      v8 = *(_QWORD *)(a1 + 32);
      v7 = *(_DWORD *)(a1 + 48);
    }
    while ( 1 )
    {
      v16 = *v9;
      if ( !v7 )
        break;
      v11 = 1;
      v12 = 0;
      v13 = (v7 - 1) & (((unsigned int)v16 >> 9) ^ ((unsigned int)v16 >> 4));
      v14 = (_QWORD *)(v8 + 16LL * v13);
      v15 = *v14;
      if ( v16 == *v14 )
      {
LABEL_6:
        if ( v10 == ++v9 )
          goto LABEL_16;
      }
      else
      {
        while ( v15 != -4096 )
        {
          if ( v12 || v15 != -8192 )
            v14 = v12;
          v13 = (v7 - 1) & (v11 + v13);
          v15 = *(_QWORD *)(v8 + 16LL * v13);
          if ( v16 == v15 )
            goto LABEL_6;
          ++v11;
          v12 = v14;
          v14 = (_QWORD *)(v8 + 16LL * v13);
        }
        if ( !v12 )
          v12 = v14;
        v35 = *(_DWORD *)(a1 + 40);
        ++*(_QWORD *)(a1 + 24);
        v21 = v35 + 1;
        if ( 4 * v21 >= 3 * v7 )
          goto LABEL_9;
        if ( v7 - (v21 + *(_DWORD *)(a1 + 44)) <= v7 >> 3 )
        {
          v91 = v6;
          sub_D39D40(v96, v7);
          v36 = *(_DWORD *)(a1 + 48);
          if ( !v36 )
          {
LABEL_147:
            ++*(_DWORD *)(a1 + 40);
            BUG();
          }
          v37 = v36 - 1;
          v38 = *(_QWORD *)(a1 + 32);
          v39 = 0;
          v40 = (v36 - 1) & (((unsigned int)v16 >> 9) ^ ((unsigned int)v16 >> 4));
          v6 = v91;
          v41 = 1;
          v21 = *(_DWORD *)(a1 + 40) + 1;
          v12 = (_QWORD *)(v38 + 16LL * v40);
          v42 = *v12;
          if ( v16 != *v12 )
          {
            while ( v42 != -4096 )
            {
              if ( !v39 && v42 == -8192 )
                v39 = v12;
              v40 = v37 & (v41 + v40);
              v12 = (_QWORD *)(v38 + 16LL * v40);
              v42 = *v12;
              if ( v16 == *v12 )
                goto LABEL_11;
              ++v41;
            }
            if ( v39 )
              v12 = v39;
          }
        }
LABEL_11:
        *(_DWORD *)(a1 + 40) = v21;
        if ( *v12 != -4096 )
          --*(_DWORD *)(a1 + 44);
        *v12 = v16;
        v23 = v99;
        *((_DWORD *)v12 + 2) = v99;
        v24 = *(_DWORD *)(a1 + 80);
        if ( !v24 )
        {
          ++*(_QWORD *)(a1 + 56);
          v102 = 0;
          goto LABEL_61;
        }
        v25 = *(_QWORD *)(a1 + 64);
        v26 = (v24 - 1) & (37 * v23);
        v27 = (int *)(v25 + 16LL * v26);
        v28 = *v27;
        if ( v23 != *v27 )
        {
          v55 = 1;
          v56 = 0;
          while ( v28 != -1 )
          {
            if ( v56 || v28 != -2 )
              v27 = v56;
            v26 = (v24 - 1) & (v55 + v26);
            v28 = *(_DWORD *)(v25 + 16LL * v26);
            if ( v23 == v28 )
              goto LABEL_15;
            ++v55;
            v56 = v27;
            v27 = (int *)(v25 + 16LL * v26);
          }
          v57 = *(_DWORD *)(a1 + 72);
          if ( !v56 )
            v56 = v27;
          ++*(_QWORD *)(a1 + 56);
          v58 = v57 + 1;
          v102 = (__int64 *)v56;
          if ( 4 * (v57 + 1) >= 3 * v24 )
          {
LABEL_61:
            v92 = v6;
            v24 *= 2;
          }
          else
          {
            if ( v24 - *(_DWORD *)(a1 + 76) - v58 > v24 >> 3 )
            {
LABEL_57:
              *(_DWORD *)(a1 + 72) = v58;
              if ( *v56 != -1 )
                --*(_DWORD *)(a1 + 76);
              *v56 = v23;
              v28 = v99;
              *((_QWORD *)v56 + 1) = v16;
              goto LABEL_15;
            }
            v92 = v6;
          }
          sub_1247200(v95, v24);
          sub_22B1B10(v95, &v99, &v102);
          v23 = v99;
          v56 = (int *)v102;
          v6 = v92;
          v58 = *(_DWORD *)(a1 + 72) + 1;
          goto LABEL_57;
        }
LABEL_15:
        ++v9;
        v8 = *(_QWORD *)(a1 + 32);
        v7 = *(_DWORD *)(a1 + 48);
        v99 = v28 + 1;
        if ( v10 == v9 )
          goto LABEL_16;
      }
    }
    ++*(_QWORD *)(a1 + 24);
LABEL_9:
    v90 = v6;
    sub_D39D40(v96, 2 * v7);
    v17 = *(_DWORD *)(a1 + 48);
    if ( !v17 )
      goto LABEL_147;
    v18 = v17 - 1;
    v19 = *(_QWORD *)(a1 + 32);
    v6 = v90;
    v20 = (v17 - 1) & (((unsigned int)v16 >> 9) ^ ((unsigned int)v16 >> 4));
    v21 = *(_DWORD *)(a1 + 40) + 1;
    v12 = (_QWORD *)(v19 + 16LL * v20);
    v22 = *v12;
    if ( v16 != *v12 )
    {
      v86 = 1;
      v87 = 0;
      while ( v22 != -4096 )
      {
        if ( v22 == -8192 && !v87 )
          v87 = v12;
        v20 = v18 & (v86 + v20);
        v12 = (_QWORD *)(v19 + 16LL * v20);
        v22 = *v12;
        if ( v16 == *v12 )
          goto LABEL_11;
        ++v86;
      }
      if ( v87 )
        v12 = v87;
    }
    goto LABEL_11;
  }
LABEL_49:
  v102 = 0;
  v103 = 0;
  *(_QWORD *)(a1 + 8) = a4;
  *(_QWORD *)(a1 + 16) = a5;
  v104 = 0;
  v105 = 0;
  sub_22B4A70(a4, a5, (__int64)&v102);
  v52 = (__int64)v103;
  v53 = (unsigned int)v105;
  if ( (_DWORD)v104 )
  {
    v62 = &v103[v53];
    if ( v103 != &v103[v53] )
    {
      v63 = v103;
      while ( *v63 == -4096 || *v63 == -8192 )
      {
        if ( v62 == ++v63 )
          return sub_C7D6A0(v52, v53 * 8, 8);
      }
      if ( v62 != v63 )
      {
        v64 = *v63;
        v65 = *(_DWORD *)(a1 + 48);
        v100 = *v63;
        if ( !v65 )
          goto LABEL_95;
        while ( 1 )
        {
          v66 = *(_QWORD *)(a1 + 32);
          v67 = 1;
          v68 = 0;
          v69 = (v65 - 1) & (((unsigned int)v64 >> 9) ^ ((unsigned int)v64 >> 4));
          v70 = (__int64 *)(v66 + 16LL * v69);
          v71 = *v70;
          if ( v64 != *v70 )
          {
            while ( v71 != -4096 )
            {
              if ( v71 != -8192 || v68 )
                v70 = (__int64 *)v68;
              v69 = (v65 - 1) & (v67 + v69);
              v71 = *(_QWORD *)(v66 + 16LL * v69);
              if ( v64 == v71 )
                goto LABEL_74;
              ++v67;
              v68 = (__int64)v70;
              v70 = (__int64 *)(v66 + 16LL * v69);
            }
            if ( !v68 )
              v68 = (__int64)v70;
            v72 = *(_DWORD *)(a1 + 40);
            ++*(_QWORD *)(a1 + 24);
            v73 = v72 + 1;
            v101 = v68;
            if ( 4 * (v72 + 1) < 3 * v65 )
            {
              v74 = v64;
              if ( v65 - *(_DWORD *)(a1 + 44) - v73 > v65 >> 3 )
                goto LABEL_88;
              goto LABEL_97;
            }
LABEL_96:
            v65 *= 2;
LABEL_97:
            sub_D39D40(v96, v65);
            sub_22B1A50(v96, &v100, &v101);
            v74 = v100;
            v68 = v101;
            v73 = *(_DWORD *)(a1 + 40) + 1;
LABEL_88:
            *(_DWORD *)(a1 + 40) = v73;
            if ( *(_QWORD *)v68 != -4096 )
              --*(_DWORD *)(a1 + 44);
            *(_QWORD *)v68 = v74;
            v75 = v99;
            *(_DWORD *)(v68 + 8) = v99;
            v76 = *(_DWORD *)(a1 + 80);
            if ( v76 )
            {
              v77 = v76 - 1;
              v78 = *(_QWORD *)(a1 + 64);
              v79 = v77 & (37 * v75);
              v80 = (_DWORD *)(v78 + 16LL * v79);
              v81 = *v80;
              if ( *v80 == v75 )
                goto LABEL_92;
              v85 = 1;
              v82 = 0;
              while ( v81 != -1 )
              {
                if ( v82 || v81 != -2 )
                  v80 = v82;
                v79 = v77 & (v85 + v79);
                v81 = *(_DWORD *)(v78 + 16LL * v79);
                if ( v75 == v81 )
                  goto LABEL_92;
                ++v85;
                v82 = v80;
                v80 = (_DWORD *)(v78 + 16LL * v79);
              }
              if ( !v82 )
                v82 = v80;
            }
            else
            {
              v82 = 0;
            }
            v83 = sub_22B3610(v95, &v99, v82);
            v75 = v99;
            *((_QWORD *)v83 + 1) = v64;
            *v83 = v75;
LABEL_92:
            v99 = v75 + 1;
          }
LABEL_74:
          if ( ++v63 == v62 )
            goto LABEL_78;
          while ( *v63 == -4096 || *v63 == -8192 )
          {
            if ( v62 == ++v63 )
              goto LABEL_78;
          }
          if ( v63 == v62 )
          {
LABEL_78:
            v52 = (__int64)v103;
            v53 = (unsigned int)v105;
            return sub_C7D6A0(v52, v53 * 8, 8);
          }
          v64 = *v63;
          v65 = *(_DWORD *)(a1 + 48);
          v100 = *v63;
          if ( !v65 )
          {
LABEL_95:
            ++*(_QWORD *)(a1 + 24);
            v101 = 0;
            goto LABEL_96;
          }
        }
      }
    }
  }
  return sub_C7D6A0(v52, v53 * 8, 8);
}
