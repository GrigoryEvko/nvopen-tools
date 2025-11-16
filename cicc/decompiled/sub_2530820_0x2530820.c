// Function: sub_2530820
// Address: 0x2530820
//
__int64 __fastcall sub_2530820(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5, _BYTE *a6, char a7)
{
  unsigned __int64 v11; // r15
  __int64 v12; // rcx
  __int64 v13; // rdx
  __int64 v14; // rax
  __int64 v15; // r8
  __int64 v16; // rax
  __int64 v17; // rax
  unsigned __int64 v18; // rax
  __int64 v19; // rdx
  __int64 v20; // rdi
  __int64 *v21; // rbx
  __int64 *v22; // r15
  __int64 v23; // r14
  __int64 v24; // rax
  __int64 *v25; // rbx
  __int64 *v26; // r15
  _QWORD *v27; // rdi
  __int64 *v28; // rsi
  __int64 v29; // r8
  __int64 v30; // r9
  int v31; // eax
  unsigned int v32; // esi
  __int64 v33; // r9
  __int64 *v34; // r11
  int v35; // r12d
  unsigned int v36; // edx
  __int64 *v37; // r8
  __int64 v38; // rdi
  int v39; // eax
  __int64 v40; // r12
  __int64 v41; // rax
  __int64 v42; // rbx
  __int64 *v43; // r12
  __int64 *v44; // r14
  _QWORD *v45; // rdi
  __int64 *v46; // rsi
  __int64 v47; // r8
  __int64 v48; // r9
  int v49; // eax
  unsigned int v50; // esi
  __int64 v51; // r9
  _QWORD *v52; // r11
  int v53; // r13d
  unsigned int v54; // edx
  _QWORD *v55; // r8
  __int64 v56; // rdi
  int v57; // eax
  __int64 *v58; // rax
  __int64 v59; // r13
  __int64 v60; // rax
  __int64 v62; // r12
  __int64 v63; // rax
  __int64 *v64; // r12
  __int64 *v65; // r13
  __int64 v66; // r8
  unsigned int v67; // eax
  __int64 *v68; // rdi
  __int64 v69; // rcx
  unsigned int v70; // esi
  __int64 *v71; // r10
  int v72; // edx
  __int64 v73; // r13
  __int64 v74; // rax
  __int64 *v75; // r13
  __int64 *v76; // r15
  __int64 v77; // r8
  unsigned int v78; // eax
  _QWORD *v79; // rdi
  __int64 v80; // rcx
  unsigned int v81; // esi
  int v82; // edx
  __int64 *v83; // rax
  __int64 v84; // rcx
  __int64 v85; // rdi
  int v86; // r10d
  unsigned int i; // eax
  __int64 v88; // rsi
  unsigned int v89; // eax
  int v90; // r11d
  int v91; // eax
  int v92; // r11d
  _QWORD *v93; // r10
  int v94; // eax
  __int64 v95; // [rsp+10h] [rbp-200h]
  __int64 v96; // [rsp+30h] [rbp-1E0h]
  unsigned __int8 v97; // [rsp+30h] [rbp-1E0h]
  const void *v99; // [rsp+38h] [rbp-1D8h]
  const void *v100; // [rsp+38h] [rbp-1D8h]
  char v101; // [rsp+44h] [rbp-1CCh] BYREF
  __int64 v102; // [rsp+48h] [rbp-1C8h] BYREF
  __int64 v103; // [rsp+50h] [rbp-1C0h] BYREF
  __int64 *v104; // [rsp+58h] [rbp-1B8h] BYREF
  __int64 *v105; // [rsp+60h] [rbp-1B0h] BYREF
  __int64 v106; // [rsp+68h] [rbp-1A8h]
  _BYTE v107[48]; // [rsp+70h] [rbp-1A0h] BYREF
  _QWORD v108[12]; // [rsp+A0h] [rbp-170h] BYREF
  __int64 v109; // [rsp+100h] [rbp-110h] BYREF
  __int64 v110; // [rsp+108h] [rbp-108h]
  __int64 v111; // [rsp+110h] [rbp-100h]
  __int64 v112; // [rsp+118h] [rbp-F8h]
  __int64 *v113; // [rsp+120h] [rbp-F0h]
  __int64 v114; // [rsp+128h] [rbp-E8h]
  _BYTE v115[64]; // [rsp+130h] [rbp-E0h] BYREF
  __int64 v116; // [rsp+170h] [rbp-A0h] BYREF
  __int64 v117; // [rsp+178h] [rbp-98h]
  __int64 v118; // [rsp+180h] [rbp-90h]
  __int64 v119; // [rsp+188h] [rbp-88h]
  __int64 *v120; // [rsp+190h] [rbp-80h]
  __int64 v121; // [rsp+198h] [rbp-78h]
  _BYTE v122[112]; // [rsp+1A0h] [rbp-70h] BYREF

  v11 = *(_QWORD *)(a2 - 32);
  v102 = a4;
  v101 = a7;
  v105 = (__int64 *)v107;
  v106 = 0x600000000LL;
  v120 = (__int64 *)v122;
  v12 = *(_QWORD *)(a1 + 208);
  v113 = (__int64 *)v115;
  v96 = v12;
  v109 = 0;
  v110 = 0;
  v111 = 0;
  v112 = 0;
  v114 = 0x800000000LL;
  v116 = 0;
  v117 = 0;
  v118 = 0;
  v119 = 0;
  v121 = 0x800000000LL;
  v13 = sub_B43CB0(a2);
  v14 = *(_QWORD *)(v96 + 240);
  v15 = *(_QWORD *)v14;
  if ( !*(_QWORD *)v14 )
    goto LABEL_91;
  if ( *(_BYTE *)(v14 + 16) )
  {
    v84 = *(unsigned int *)(v15 + 88);
    v85 = *(_QWORD *)(v15 + 72);
    if ( (_DWORD)v84 )
    {
      v86 = 1;
      for ( i = (v84 - 1)
              & (((0xBF58476D1CE4E5B9LL
                 * (((unsigned int)v13 >> 9) ^ ((unsigned int)v13 >> 4)
                  | ((unsigned __int64)(((unsigned int)&unk_4F6D3F8 >> 9) ^ ((unsigned int)&unk_4F6D3F8 >> 4)) << 32))) >> 31)
               ^ (484763065 * (((unsigned int)v13 >> 9) ^ ((unsigned int)v13 >> 4)))); ; i = (v84 - 1) & v89 )
      {
        v88 = v85 + 24LL * i;
        if ( *(_UNKNOWN **)v88 == &unk_4F6D3F8 && v13 == *(_QWORD *)(v88 + 8) )
          break;
        if ( *(_QWORD *)v88 == -4096 && *(_QWORD *)(v88 + 8) == -4096 )
        {
          v17 = 0;
          goto LABEL_5;
        }
        v89 = v86 + i;
        ++v86;
      }
      if ( v88 != v85 + 24 * v84 )
      {
        v16 = *(_QWORD *)(*(_QWORD *)(v88 + 16) + 24LL);
        if ( v16 )
          goto LABEL_4;
      }
    }
LABEL_91:
    v17 = 0;
    goto LABEL_5;
  }
  v16 = sub_BC1CD0(*(_QWORD *)v14, &unk_4F6D3F8, v13);
LABEL_4:
  v17 = v16 + 8;
LABEL_5:
  v103 = v17;
  v108[0] = a2;
  v108[4] = a6;
  v108[5] = &v103;
  v108[6] = &v102;
  v108[7] = &v109;
  v108[8] = &v116;
  v108[9] = &v101;
  v108[1] = v11;
  v108[2] = a1;
  v108[3] = a5;
  v108[10] = &v105;
  v18 = sub_250D2C0(v11, 0);
  v20 = sub_252AE70(a1, v18, v19, a5, 1, 0, 1);
  if ( v20
    && (v97 = (*(__int64 (__fastcall **)(__int64, __int64 (__fastcall *)(__int64 *, unsigned __int8 *), _QWORD *, __int64))(*(_QWORD *)v20 + 112LL))(
                v20,
                sub_252ED40,
                v108,
                2)) != 0 )
  {
    v21 = v105;
    if ( &v105[(unsigned int)v106] != v105 )
    {
      v95 = a3;
      v22 = &v105[(unsigned int)v106];
      do
      {
        v23 = *v21;
        v24 = (*(__int64 (__fastcall **)(__int64))(*(_QWORD *)*v21 + 48LL))(*v21);
        if ( !(*(unsigned __int8 (__fastcall **)(__int64))(*(_QWORD *)v24 + 24LL))(v24) )
          *a6 = 1;
        sub_250ED80(a1, v23, a5, 1);
        ++v21;
      }
      while ( v22 != v21 );
      a3 = v95;
    }
    v25 = v113;
    v26 = &v113[(unsigned int)v114];
    if ( v113 != v26 )
    {
      v99 = (const void *)(a3 + 48);
      while ( 1 )
      {
        v31 = *(_DWORD *)(a3 + 16);
        if ( !v31 )
          break;
        v32 = *(_DWORD *)(a3 + 24);
        if ( !v32 )
        {
          ++*(_QWORD *)a3;
          v104 = 0;
LABEL_107:
          v32 *= 2;
LABEL_108:
          sub_CE2A30(a3, v32);
          sub_DA5B20(a3, v25, &v104);
          v34 = v104;
          v39 = *(_DWORD *)(a3 + 16) + 1;
          goto LABEL_25;
        }
        v33 = *(_QWORD *)(a3 + 8);
        v34 = 0;
        v35 = 1;
        v36 = (v32 - 1) & (((unsigned int)*v25 >> 9) ^ ((unsigned int)*v25 >> 4));
        v37 = (__int64 *)(v33 + 8LL * v36);
        v38 = *v37;
        if ( *v25 == *v37 )
        {
LABEL_16:
          if ( ++v25 == v26 )
            goto LABEL_30;
        }
        else
        {
          while ( v38 != -4096 )
          {
            if ( v38 != -8192 || v34 )
              v37 = v34;
            v36 = (v32 - 1) & (v35 + v36);
            v38 = *(_QWORD *)(v33 + 8LL * v36);
            if ( *v25 == v38 )
              goto LABEL_16;
            ++v35;
            v34 = v37;
            v37 = (__int64 *)(v33 + 8LL * v36);
          }
          if ( !v34 )
            v34 = v37;
          v39 = v31 + 1;
          ++*(_QWORD *)a3;
          v104 = v34;
          if ( 4 * v39 >= 3 * v32 )
            goto LABEL_107;
          if ( v32 - *(_DWORD *)(a3 + 20) - v39 <= v32 >> 3 )
            goto LABEL_108;
LABEL_25:
          *(_DWORD *)(a3 + 16) = v39;
          if ( *v34 != -4096 )
            --*(_DWORD *)(a3 + 20);
          v40 = *v25;
          *v34 = *v25;
          v41 = *(unsigned int *)(a3 + 40);
          if ( v41 + 1 > (unsigned __int64)*(unsigned int *)(a3 + 44) )
          {
            sub_C8D5F0(a3 + 32, v99, v41 + 1, 8u, (__int64)v37, v33);
            v41 = *(unsigned int *)(a3 + 40);
          }
          ++v25;
          *(_QWORD *)(*(_QWORD *)(a3 + 32) + 8 * v41) = v40;
          ++*(_DWORD *)(a3 + 40);
          if ( v25 == v26 )
            goto LABEL_30;
        }
      }
      v27 = *(_QWORD **)(a3 + 32);
      v28 = &v27[*(unsigned int *)(a3 + 40)];
      if ( v28 != sub_2506440(v27, (__int64)v28, v25) )
        goto LABEL_16;
      v62 = *v25;
      if ( v29 + 1 > (unsigned __int64)*(unsigned int *)(a3 + 44) )
      {
        sub_C8D5F0(a3 + 32, v99, v29 + 1, 8u, v29, v30);
        v28 = (__int64 *)(*(_QWORD *)(a3 + 32) + 8LL * *(unsigned int *)(a3 + 40));
      }
      *v28 = v62;
      v63 = (unsigned int)(*(_DWORD *)(a3 + 40) + 1);
      *(_DWORD *)(a3 + 40) = v63;
      if ( (unsigned int)v63 <= 4 )
        goto LABEL_16;
      v64 = *(__int64 **)(a3 + 32);
      v65 = &v64[v63];
      while ( 1 )
      {
        v70 = *(_DWORD *)(a3 + 24);
        if ( !v70 )
          break;
        v66 = *(_QWORD *)(a3 + 8);
        v67 = (v70 - 1) & (((unsigned int)*v64 >> 9) ^ ((unsigned int)*v64 >> 4));
        v68 = (__int64 *)(v66 + 8LL * v67);
        v69 = *v68;
        if ( *v68 != *v64 )
        {
          v90 = 1;
          v71 = 0;
          while ( v69 != -4096 )
          {
            if ( v71 || v69 != -8192 )
              v68 = v71;
            v67 = (v70 - 1) & (v90 + v67);
            v69 = *(_QWORD *)(v66 + 8LL * v67);
            if ( *v64 == v69 )
              goto LABEL_61;
            ++v90;
            v71 = v68;
            v68 = (__int64 *)(v66 + 8LL * v67);
          }
          v91 = *(_DWORD *)(a3 + 16);
          if ( !v71 )
            v71 = v68;
          ++*(_QWORD *)a3;
          v72 = v91 + 1;
          v104 = v71;
          if ( 4 * (v91 + 1) < 3 * v70 )
          {
            if ( v70 - *(_DWORD *)(a3 + 20) - v72 <= v70 >> 3 )
            {
LABEL_65:
              sub_CE2A30(a3, v70);
              sub_DA5B20(a3, v64, &v104);
              v71 = v104;
              v72 = *(_DWORD *)(a3 + 16) + 1;
            }
            *(_DWORD *)(a3 + 16) = v72;
            if ( *v71 != -4096 )
              --*(_DWORD *)(a3 + 20);
            *v71 = *v64;
            goto LABEL_61;
          }
LABEL_64:
          v70 *= 2;
          goto LABEL_65;
        }
LABEL_61:
        if ( v65 == ++v64 )
          goto LABEL_16;
      }
      ++*(_QWORD *)a3;
      v104 = 0;
      goto LABEL_64;
    }
LABEL_30:
    v42 = v102;
    v43 = v120;
    if ( v102 )
    {
      v44 = &v120[(unsigned int)v121];
      if ( v44 != v120 )
      {
        v100 = (const void *)(v102 + 48);
        while ( 1 )
        {
          v49 = *(_DWORD *)(v42 + 16);
          if ( !v49 )
            break;
          v50 = *(_DWORD *)(v42 + 24);
          if ( !v50 )
          {
            v104 = 0;
            ++*(_QWORD *)v42;
LABEL_110:
            v50 *= 2;
LABEL_111:
            sub_CF4090(v42, v50);
            sub_23FDF60(v42, v43, &v104);
            v57 = *(_DWORD *)(v42 + 16) + 1;
            goto LABEL_43;
          }
          v51 = *(_QWORD *)(v42 + 8);
          v52 = 0;
          v53 = 1;
          v54 = (v50 - 1) & (((unsigned int)*v43 >> 9) ^ ((unsigned int)*v43 >> 4));
          v55 = (_QWORD *)(v51 + 8LL * v54);
          v56 = *v55;
          if ( *v43 == *v55 )
          {
LABEL_34:
            if ( ++v43 == v44 )
              goto LABEL_48;
          }
          else
          {
            while ( v56 != -4096 )
            {
              if ( v56 != -8192 || v52 )
                v55 = v52;
              v54 = (v50 - 1) & (v53 + v54);
              v56 = *(_QWORD *)(v51 + 8LL * v54);
              if ( *v43 == v56 )
                goto LABEL_34;
              ++v53;
              v52 = v55;
              v55 = (_QWORD *)(v51 + 8LL * v54);
            }
            if ( !v52 )
              v52 = v55;
            v57 = v49 + 1;
            v104 = v52;
            ++*(_QWORD *)v42;
            if ( 4 * v57 >= 3 * v50 )
              goto LABEL_110;
            if ( v50 - *(_DWORD *)(v42 + 20) - v57 <= v50 >> 3 )
              goto LABEL_111;
LABEL_43:
            *(_DWORD *)(v42 + 16) = v57;
            v58 = v104;
            if ( *v104 != -4096 )
              --*(_DWORD *)(v42 + 20);
            v59 = *v43;
            *v58 = *v43;
            v60 = *(unsigned int *)(v42 + 40);
            if ( v60 + 1 > (unsigned __int64)*(unsigned int *)(v42 + 44) )
            {
              sub_C8D5F0(v42 + 32, v100, v60 + 1, 8u, (__int64)v55, v51);
              v60 = *(unsigned int *)(v42 + 40);
            }
            ++v43;
            *(_QWORD *)(*(_QWORD *)(v42 + 32) + 8 * v60) = v59;
            ++*(_DWORD *)(v42 + 40);
            if ( v43 == v44 )
            {
LABEL_48:
              v43 = v120;
              goto LABEL_49;
            }
          }
        }
        v45 = *(_QWORD **)(v42 + 32);
        v46 = &v45[*(unsigned int *)(v42 + 40)];
        if ( v46 != sub_2506500(v45, (__int64)v46, v43) )
          goto LABEL_34;
        v73 = *v43;
        if ( v47 + 1 > (unsigned __int64)*(unsigned int *)(v42 + 44) )
        {
          sub_C8D5F0(v42 + 32, v100, v47 + 1, 8u, v47, v48);
          v46 = (__int64 *)(*(_QWORD *)(v42 + 32) + 8LL * *(unsigned int *)(v42 + 40));
        }
        *v46 = v73;
        v74 = (unsigned int)(*(_DWORD *)(v42 + 40) + 1);
        *(_DWORD *)(v42 + 40) = v74;
        if ( (unsigned int)v74 <= 4 )
          goto LABEL_34;
        v75 = *(__int64 **)(v42 + 32);
        v76 = &v75[v74];
        while ( 1 )
        {
          v81 = *(_DWORD *)(v42 + 24);
          if ( !v81 )
            break;
          v77 = *(_QWORD *)(v42 + 8);
          v78 = (v81 - 1) & (((unsigned int)*v75 >> 9) ^ ((unsigned int)*v75 >> 4));
          v79 = (_QWORD *)(v77 + 8LL * v78);
          v80 = *v79;
          if ( *v79 != *v75 )
          {
            v92 = 1;
            v93 = 0;
            while ( v80 != -4096 )
            {
              if ( v93 || v80 != -8192 )
                v79 = v93;
              v78 = (v81 - 1) & (v92 + v78);
              v80 = *(_QWORD *)(v77 + 8LL * v78);
              if ( *v75 == v80 )
                goto LABEL_74;
              ++v92;
              v93 = v79;
              v79 = (_QWORD *)(v77 + 8LL * v78);
            }
            if ( !v93 )
              v93 = v79;
            v104 = v93;
            v94 = *(_DWORD *)(v42 + 16);
            ++*(_QWORD *)v42;
            v82 = v94 + 1;
            if ( 4 * (v94 + 1) < 3 * v81 )
            {
              if ( v81 - *(_DWORD *)(v42 + 20) - v82 <= v81 >> 3 )
              {
LABEL_78:
                sub_CF4090(v42, v81);
                sub_23FDF60(v42, v75, &v104);
                v82 = *(_DWORD *)(v42 + 16) + 1;
              }
              *(_DWORD *)(v42 + 16) = v82;
              v83 = v104;
              if ( *v104 != -4096 )
                --*(_DWORD *)(v42 + 20);
              *v83 = *v75;
              goto LABEL_74;
            }
LABEL_77:
            v81 *= 2;
            goto LABEL_78;
          }
LABEL_74:
          if ( v76 == ++v75 )
            goto LABEL_34;
        }
        v104 = 0;
        ++*(_QWORD *)v42;
        goto LABEL_77;
      }
    }
  }
  else
  {
    v97 = 0;
    v43 = v120;
  }
LABEL_49:
  if ( v43 != (__int64 *)v122 )
    _libc_free((unsigned __int64)v43);
  sub_C7D6A0(v117, 8LL * (unsigned int)v119, 8);
  if ( v113 != (__int64 *)v115 )
    _libc_free((unsigned __int64)v113);
  sub_C7D6A0(v110, 8LL * (unsigned int)v112, 8);
  if ( v105 != (__int64 *)v107 )
    _libc_free((unsigned __int64)v105);
  return v97;
}
