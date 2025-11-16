// Function: sub_252F460
// Address: 0x252f460
//
__int64 __fastcall sub_252F460(__int64 a1, __int64 a2, __int64 a3, __int64 a4, _BYTE *a5, char a6)
{
  unsigned __int64 v10; // r15
  __int64 v11; // rcx
  __int64 v12; // rdx
  __int64 v13; // rax
  __int64 v14; // r8
  __int64 v15; // rax
  __int64 v16; // rax
  unsigned __int64 v17; // rax
  __int64 v18; // rdx
  __int64 v19; // rdi
  __int64 *v20; // rbx
  __int64 *v21; // r15
  __int64 v22; // r14
  __int64 v23; // rax
  __int64 *v24; // rbx
  __int64 *v25; // r15
  _QWORD *v26; // rdi
  __int64 *v27; // rsi
  __int64 v28; // r8
  __int64 v29; // r9
  int v30; // eax
  unsigned int v31; // esi
  __int64 v32; // r9
  __int64 *v33; // r11
  int v34; // r12d
  unsigned int v35; // edx
  __int64 *v36; // r8
  __int64 v37; // rdi
  int v38; // eax
  __int64 v39; // r12
  __int64 v40; // rax
  __int64 v41; // rbx
  __int64 *v42; // r12
  __int64 *v43; // r14
  _QWORD *v44; // rdi
  __int64 *v45; // rsi
  __int64 v46; // r8
  __int64 v47; // r9
  int v48; // eax
  unsigned int v49; // esi
  __int64 v50; // r9
  _QWORD *v51; // r11
  int v52; // r13d
  unsigned int v53; // edx
  _QWORD *v54; // r8
  __int64 v55; // rdi
  int v56; // eax
  __int64 *v57; // rax
  __int64 v58; // r13
  __int64 v59; // rax
  __int64 v61; // r12
  __int64 v62; // rax
  __int64 *v63; // r12
  __int64 *v64; // r13
  __int64 v65; // r8
  unsigned int v66; // eax
  __int64 *v67; // rdi
  __int64 v68; // rcx
  unsigned int v69; // esi
  __int64 *v70; // r10
  int v71; // edx
  __int64 v72; // r13
  __int64 v73; // rax
  __int64 *v74; // r13
  __int64 *v75; // r15
  __int64 v76; // r8
  unsigned int v77; // eax
  _QWORD *v78; // rdi
  __int64 v79; // rcx
  unsigned int v80; // esi
  int v81; // edx
  __int64 *v82; // rax
  __int64 v83; // rcx
  __int64 v84; // rdi
  int v85; // r10d
  unsigned int i; // eax
  __int64 v87; // rsi
  unsigned int v88; // eax
  int v89; // r11d
  int v90; // eax
  int v91; // r11d
  _QWORD *v92; // r10
  int v93; // eax
  __int64 v94; // [rsp+8h] [rbp-208h]
  __int64 v95; // [rsp+28h] [rbp-1E8h]
  unsigned __int8 v96; // [rsp+28h] [rbp-1E8h]
  const void *v98; // [rsp+30h] [rbp-1E0h]
  const void *v99; // [rsp+30h] [rbp-1E0h]
  char v100; // [rsp+3Ch] [rbp-1D4h] BYREF
  __int64 v101; // [rsp+48h] [rbp-1C8h] BYREF
  __int64 v102; // [rsp+50h] [rbp-1C0h] BYREF
  __int64 *v103; // [rsp+58h] [rbp-1B8h] BYREF
  __int64 *v104; // [rsp+60h] [rbp-1B0h] BYREF
  __int64 v105; // [rsp+68h] [rbp-1A8h]
  _BYTE v106[48]; // [rsp+70h] [rbp-1A0h] BYREF
  _QWORD v107[12]; // [rsp+A0h] [rbp-170h] BYREF
  __int64 v108; // [rsp+100h] [rbp-110h] BYREF
  __int64 v109; // [rsp+108h] [rbp-108h]
  __int64 v110; // [rsp+110h] [rbp-100h]
  __int64 v111; // [rsp+118h] [rbp-F8h]
  __int64 *v112; // [rsp+120h] [rbp-F0h]
  __int64 v113; // [rsp+128h] [rbp-E8h]
  _BYTE v114[64]; // [rsp+130h] [rbp-E0h] BYREF
  __int64 v115; // [rsp+170h] [rbp-A0h] BYREF
  __int64 v116; // [rsp+178h] [rbp-98h]
  __int64 v117; // [rsp+180h] [rbp-90h]
  __int64 v118; // [rsp+188h] [rbp-88h]
  __int64 *v119; // [rsp+190h] [rbp-80h]
  __int64 v120; // [rsp+198h] [rbp-78h]
  _BYTE v121[112]; // [rsp+1A0h] [rbp-70h] BYREF

  v119 = (__int64 *)v121;
  v10 = *(_QWORD *)(a2 - 32);
  v104 = (__int64 *)v106;
  v105 = 0x600000000LL;
  v11 = *(_QWORD *)(a1 + 208);
  v112 = (__int64 *)v114;
  v95 = v11;
  v100 = a6;
  v101 = 0;
  v108 = 0;
  v109 = 0;
  v110 = 0;
  v111 = 0;
  v113 = 0x800000000LL;
  v115 = 0;
  v116 = 0;
  v117 = 0;
  v118 = 0;
  v120 = 0x800000000LL;
  v12 = sub_B43CB0(a2);
  v13 = *(_QWORD *)(v95 + 240);
  v14 = *(_QWORD *)v13;
  if ( !*(_QWORD *)v13 )
    goto LABEL_91;
  if ( *(_BYTE *)(v13 + 16) )
  {
    v83 = *(unsigned int *)(v14 + 88);
    v84 = *(_QWORD *)(v14 + 72);
    if ( (_DWORD)v83 )
    {
      v85 = 1;
      for ( i = (v83 - 1)
              & (((0xBF58476D1CE4E5B9LL
                 * (((unsigned int)v12 >> 9) ^ ((unsigned int)v12 >> 4)
                  | ((unsigned __int64)(((unsigned int)&unk_4F6D3F8 >> 9) ^ ((unsigned int)&unk_4F6D3F8 >> 4)) << 32))) >> 31)
               ^ (484763065 * (((unsigned int)v12 >> 9) ^ ((unsigned int)v12 >> 4)))); ; i = (v83 - 1) & v88 )
      {
        v87 = v84 + 24LL * i;
        if ( *(_UNKNOWN **)v87 == &unk_4F6D3F8 && v12 == *(_QWORD *)(v87 + 8) )
          break;
        if ( *(_QWORD *)v87 == -4096 && *(_QWORD *)(v87 + 8) == -4096 )
        {
          v16 = 0;
          goto LABEL_5;
        }
        v88 = v85 + i;
        ++v85;
      }
      if ( v87 != v84 + 24 * v83 )
      {
        v15 = *(_QWORD *)(*(_QWORD *)(v87 + 16) + 24LL);
        if ( v15 )
          goto LABEL_4;
      }
    }
LABEL_91:
    v16 = 0;
    goto LABEL_5;
  }
  v15 = sub_BC1CD0(*(_QWORD *)v13, &unk_4F6D3F8, v12);
LABEL_4:
  v16 = v15 + 8;
LABEL_5:
  v102 = v16;
  v107[0] = a2;
  v107[4] = a5;
  v107[5] = &v102;
  v107[6] = &v101;
  v107[7] = &v108;
  v107[8] = &v115;
  v107[9] = &v100;
  v107[1] = v10;
  v107[2] = a1;
  v107[3] = a4;
  v107[10] = &v104;
  v17 = sub_250D2C0(v10, 0);
  v19 = sub_252AE70(a1, v17, v18, a4, 1, 0, 1);
  if ( v19
    && (v96 = (*(__int64 (__fastcall **)(__int64, __int64 (__fastcall *)(__int64 *, unsigned __int8 *), _QWORD *, __int64))(*(_QWORD *)v19 + 112LL))(
                v19,
                sub_252AB80,
                v107,
                2)) != 0 )
  {
    v20 = v104;
    if ( v104 != &v104[(unsigned int)v105] )
    {
      v94 = a3;
      v21 = &v104[(unsigned int)v105];
      do
      {
        v22 = *v20;
        v23 = (*(__int64 (__fastcall **)(__int64))(*(_QWORD *)*v20 + 48LL))(*v20);
        if ( !(*(unsigned __int8 (__fastcall **)(__int64))(*(_QWORD *)v23 + 24LL))(v23) )
          *a5 = 1;
        sub_250ED80(a1, v22, a4, 1);
        ++v20;
      }
      while ( v21 != v20 );
      a3 = v94;
    }
    v24 = v112;
    v25 = &v112[(unsigned int)v113];
    if ( v112 != v25 )
    {
      v98 = (const void *)(a3 + 48);
      while ( 1 )
      {
        v30 = *(_DWORD *)(a3 + 16);
        if ( !v30 )
          break;
        v31 = *(_DWORD *)(a3 + 24);
        if ( !v31 )
        {
          ++*(_QWORD *)a3;
          v103 = 0;
LABEL_107:
          v31 *= 2;
LABEL_108:
          sub_CE2A30(a3, v31);
          sub_DA5B20(a3, v24, &v103);
          v33 = v103;
          v38 = *(_DWORD *)(a3 + 16) + 1;
          goto LABEL_25;
        }
        v32 = *(_QWORD *)(a3 + 8);
        v33 = 0;
        v34 = 1;
        v35 = (v31 - 1) & (((unsigned int)*v24 >> 9) ^ ((unsigned int)*v24 >> 4));
        v36 = (__int64 *)(v32 + 8LL * v35);
        v37 = *v36;
        if ( *v24 == *v36 )
        {
LABEL_16:
          if ( v25 == ++v24 )
            goto LABEL_30;
        }
        else
        {
          while ( v37 != -4096 )
          {
            if ( v37 != -8192 || v33 )
              v36 = v33;
            v35 = (v31 - 1) & (v34 + v35);
            v37 = *(_QWORD *)(v32 + 8LL * v35);
            if ( *v24 == v37 )
              goto LABEL_16;
            ++v34;
            v33 = v36;
            v36 = (__int64 *)(v32 + 8LL * v35);
          }
          if ( !v33 )
            v33 = v36;
          v38 = v30 + 1;
          ++*(_QWORD *)a3;
          v103 = v33;
          if ( 4 * v38 >= 3 * v31 )
            goto LABEL_107;
          if ( v31 - *(_DWORD *)(a3 + 20) - v38 <= v31 >> 3 )
            goto LABEL_108;
LABEL_25:
          *(_DWORD *)(a3 + 16) = v38;
          if ( *v33 != -4096 )
            --*(_DWORD *)(a3 + 20);
          v39 = *v24;
          *v33 = *v24;
          v40 = *(unsigned int *)(a3 + 40);
          if ( v40 + 1 > (unsigned __int64)*(unsigned int *)(a3 + 44) )
          {
            sub_C8D5F0(a3 + 32, v98, v40 + 1, 8u, (__int64)v36, v32);
            v40 = *(unsigned int *)(a3 + 40);
          }
          ++v24;
          *(_QWORD *)(*(_QWORD *)(a3 + 32) + 8 * v40) = v39;
          ++*(_DWORD *)(a3 + 40);
          if ( v25 == v24 )
            goto LABEL_30;
        }
      }
      v26 = *(_QWORD **)(a3 + 32);
      v27 = &v26[*(unsigned int *)(a3 + 40)];
      if ( v27 != sub_2506440(v26, (__int64)v27, v24) )
        goto LABEL_16;
      v61 = *v24;
      if ( v28 + 1 > (unsigned __int64)*(unsigned int *)(a3 + 44) )
      {
        sub_C8D5F0(a3 + 32, v98, v28 + 1, 8u, v28, v29);
        v27 = (__int64 *)(*(_QWORD *)(a3 + 32) + 8LL * *(unsigned int *)(a3 + 40));
      }
      *v27 = v61;
      v62 = (unsigned int)(*(_DWORD *)(a3 + 40) + 1);
      *(_DWORD *)(a3 + 40) = v62;
      if ( (unsigned int)v62 <= 4 )
        goto LABEL_16;
      v63 = *(__int64 **)(a3 + 32);
      v64 = &v63[v62];
      while ( 1 )
      {
        v69 = *(_DWORD *)(a3 + 24);
        if ( !v69 )
          break;
        v65 = *(_QWORD *)(a3 + 8);
        v66 = (v69 - 1) & (((unsigned int)*v63 >> 9) ^ ((unsigned int)*v63 >> 4));
        v67 = (__int64 *)(v65 + 8LL * v66);
        v68 = *v67;
        if ( *v67 != *v63 )
        {
          v89 = 1;
          v70 = 0;
          while ( v68 != -4096 )
          {
            if ( v70 || v68 != -8192 )
              v67 = v70;
            v66 = (v69 - 1) & (v89 + v66);
            v68 = *(_QWORD *)(v65 + 8LL * v66);
            if ( *v63 == v68 )
              goto LABEL_61;
            ++v89;
            v70 = v67;
            v67 = (__int64 *)(v65 + 8LL * v66);
          }
          v90 = *(_DWORD *)(a3 + 16);
          if ( !v70 )
            v70 = v67;
          ++*(_QWORD *)a3;
          v71 = v90 + 1;
          v103 = v70;
          if ( 4 * (v90 + 1) < 3 * v69 )
          {
            if ( v69 - *(_DWORD *)(a3 + 20) - v71 <= v69 >> 3 )
            {
LABEL_65:
              sub_CE2A30(a3, v69);
              sub_DA5B20(a3, v63, &v103);
              v70 = v103;
              v71 = *(_DWORD *)(a3 + 16) + 1;
            }
            *(_DWORD *)(a3 + 16) = v71;
            if ( *v70 != -4096 )
              --*(_DWORD *)(a3 + 20);
            *v70 = *v63;
            goto LABEL_61;
          }
LABEL_64:
          v69 *= 2;
          goto LABEL_65;
        }
LABEL_61:
        if ( v64 == ++v63 )
          goto LABEL_16;
      }
      ++*(_QWORD *)a3;
      v103 = 0;
      goto LABEL_64;
    }
LABEL_30:
    v41 = v101;
    v42 = v119;
    if ( v101 )
    {
      v43 = &v119[(unsigned int)v120];
      if ( v43 != v119 )
      {
        v99 = (const void *)(v101 + 48);
        while ( 1 )
        {
          v48 = *(_DWORD *)(v41 + 16);
          if ( !v48 )
            break;
          v49 = *(_DWORD *)(v41 + 24);
          if ( !v49 )
          {
            v103 = 0;
            ++*(_QWORD *)v41;
LABEL_110:
            v49 *= 2;
LABEL_111:
            sub_CF4090(v41, v49);
            sub_23FDF60(v41, v42, &v103);
            v56 = *(_DWORD *)(v41 + 16) + 1;
            goto LABEL_43;
          }
          v50 = *(_QWORD *)(v41 + 8);
          v51 = 0;
          v52 = 1;
          v53 = (v49 - 1) & (((unsigned int)*v42 >> 9) ^ ((unsigned int)*v42 >> 4));
          v54 = (_QWORD *)(v50 + 8LL * v53);
          v55 = *v54;
          if ( *v42 == *v54 )
          {
LABEL_34:
            if ( v43 == ++v42 )
              goto LABEL_48;
          }
          else
          {
            while ( v55 != -4096 )
            {
              if ( v55 != -8192 || v51 )
                v54 = v51;
              v53 = (v49 - 1) & (v52 + v53);
              v55 = *(_QWORD *)(v50 + 8LL * v53);
              if ( *v42 == v55 )
                goto LABEL_34;
              ++v52;
              v51 = v54;
              v54 = (_QWORD *)(v50 + 8LL * v53);
            }
            if ( !v51 )
              v51 = v54;
            v56 = v48 + 1;
            v103 = v51;
            ++*(_QWORD *)v41;
            if ( 4 * v56 >= 3 * v49 )
              goto LABEL_110;
            if ( v49 - *(_DWORD *)(v41 + 20) - v56 <= v49 >> 3 )
              goto LABEL_111;
LABEL_43:
            *(_DWORD *)(v41 + 16) = v56;
            v57 = v103;
            if ( *v103 != -4096 )
              --*(_DWORD *)(v41 + 20);
            v58 = *v42;
            *v57 = *v42;
            v59 = *(unsigned int *)(v41 + 40);
            if ( v59 + 1 > (unsigned __int64)*(unsigned int *)(v41 + 44) )
            {
              sub_C8D5F0(v41 + 32, v99, v59 + 1, 8u, (__int64)v54, v50);
              v59 = *(unsigned int *)(v41 + 40);
            }
            ++v42;
            *(_QWORD *)(*(_QWORD *)(v41 + 32) + 8 * v59) = v58;
            ++*(_DWORD *)(v41 + 40);
            if ( v43 == v42 )
            {
LABEL_48:
              v42 = v119;
              goto LABEL_49;
            }
          }
        }
        v44 = *(_QWORD **)(v41 + 32);
        v45 = &v44[*(unsigned int *)(v41 + 40)];
        if ( v45 != sub_2506500(v44, (__int64)v45, v42) )
          goto LABEL_34;
        v72 = *v42;
        if ( v46 + 1 > (unsigned __int64)*(unsigned int *)(v41 + 44) )
        {
          sub_C8D5F0(v41 + 32, v99, v46 + 1, 8u, v46, v47);
          v45 = (__int64 *)(*(_QWORD *)(v41 + 32) + 8LL * *(unsigned int *)(v41 + 40));
        }
        *v45 = v72;
        v73 = (unsigned int)(*(_DWORD *)(v41 + 40) + 1);
        *(_DWORD *)(v41 + 40) = v73;
        if ( (unsigned int)v73 <= 4 )
          goto LABEL_34;
        v74 = *(__int64 **)(v41 + 32);
        v75 = &v74[v73];
        while ( 1 )
        {
          v80 = *(_DWORD *)(v41 + 24);
          if ( !v80 )
            break;
          v76 = *(_QWORD *)(v41 + 8);
          v77 = (v80 - 1) & (((unsigned int)*v74 >> 9) ^ ((unsigned int)*v74 >> 4));
          v78 = (_QWORD *)(v76 + 8LL * v77);
          v79 = *v78;
          if ( *v78 != *v74 )
          {
            v91 = 1;
            v92 = 0;
            while ( v79 != -4096 )
            {
              if ( v92 || v79 != -8192 )
                v78 = v92;
              v77 = (v80 - 1) & (v91 + v77);
              v79 = *(_QWORD *)(v76 + 8LL * v77);
              if ( *v74 == v79 )
                goto LABEL_74;
              ++v91;
              v92 = v78;
              v78 = (_QWORD *)(v76 + 8LL * v77);
            }
            if ( !v92 )
              v92 = v78;
            v103 = v92;
            v93 = *(_DWORD *)(v41 + 16);
            ++*(_QWORD *)v41;
            v81 = v93 + 1;
            if ( 4 * (v93 + 1) < 3 * v80 )
            {
              if ( v80 - *(_DWORD *)(v41 + 20) - v81 <= v80 >> 3 )
              {
LABEL_78:
                sub_CF4090(v41, v80);
                sub_23FDF60(v41, v74, &v103);
                v81 = *(_DWORD *)(v41 + 16) + 1;
              }
              *(_DWORD *)(v41 + 16) = v81;
              v82 = v103;
              if ( *v103 != -4096 )
                --*(_DWORD *)(v41 + 20);
              *v82 = *v74;
              goto LABEL_74;
            }
LABEL_77:
            v80 *= 2;
            goto LABEL_78;
          }
LABEL_74:
          if ( v75 == ++v74 )
            goto LABEL_34;
        }
        v103 = 0;
        ++*(_QWORD *)v41;
        goto LABEL_77;
      }
    }
  }
  else
  {
    v96 = 0;
    v42 = v119;
  }
LABEL_49:
  if ( v42 != (__int64 *)v121 )
    _libc_free((unsigned __int64)v42);
  sub_C7D6A0(v116, 8LL * (unsigned int)v118, 8);
  if ( v112 != (__int64 *)v114 )
    _libc_free((unsigned __int64)v112);
  sub_C7D6A0(v109, 8LL * (unsigned int)v111, 8);
  if ( v104 != (__int64 *)v106 )
    _libc_free((unsigned __int64)v104);
  return v96;
}
