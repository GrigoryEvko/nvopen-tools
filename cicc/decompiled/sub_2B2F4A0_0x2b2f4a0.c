// Function: sub_2B2F4A0
// Address: 0x2b2f4a0
//
_QWORD *__fastcall sub_2B2F4A0(_QWORD *a1, __int64 *a2, __int64 *a3, __int64 a4, __int64 a5, __int64 a6)
{
  char *v7; // rsi
  unsigned int v8; // eax
  unsigned int v9; // ebx
  unsigned __int64 v10; // r12
  _QWORD *i; // rax
  __int64 v12; // r12
  __int64 v13; // rdx
  __int64 *v14; // rbx
  __int64 v15; // r15
  __int64 v16; // r12
  __int64 v17; // rsi
  __int64 v18; // r9
  __int64 v19; // r13
  unsigned __int64 v20; // r8
  unsigned __int64 *v21; // rax
  int v22; // ecx
  unsigned __int64 *v23; // rdx
  __int64 v24; // rax
  unsigned __int8 *v25; // r12
  __int64 v26; // rsi
  __int64 v27; // rax
  int v28; // edi
  unsigned int v29; // edx
  unsigned __int8 **v30; // rcx
  unsigned __int8 *v31; // r8
  unsigned __int8 **v32; // rax
  unsigned __int8 *v33; // r13
  unsigned int v34; // ecx
  unsigned __int8 **v35; // rdx
  unsigned __int8 *v36; // r9
  __int64 v37; // r9
  __int64 *v38; // r9
  _QWORD *v39; // rax
  __int64 *v40; // r9
  __int64 v41; // r15
  __int64 v42; // rax
  __int64 v43; // rsi
  __int64 v44; // r12
  __int64 v45; // rbx
  __int64 v46; // rdx
  unsigned int v47; // esi
  unsigned __int8 ****v48; // r10
  __int64 v49; // r11
  char v50; // r9
  __int64 v51; // rax
  unsigned int v52; // r15d
  _BYTE **v53; // rax
  _BYTE **v54; // rcx
  unsigned __int8 *v55; // rax
  unsigned __int8 *v56; // r12
  __int64 v57; // rax
  _QWORD *v58; // rax
  __int64 *v59; // rdx
  __int64 *v60; // rax
  unsigned __int64 v62; // rsi
  int v63; // edx
  int v64; // r8d
  int v65; // ecx
  __int64 v66; // rdi
  bool v67; // al
  __int64 v68; // rdx
  __int64 v69; // rdi
  bool v70; // al
  unsigned __int8 **v71; // rdx
  char v72; // al
  __int64 v73; // rdi
  bool v74; // al
  __int64 v75; // rdi
  bool v76; // al
  __int64 v77; // rdx
  __int64 v78; // rcx
  __int64 v79; // r8
  __int64 v80; // rsi
  _BYTE *v81; // rdi
  bool v82; // al
  unsigned __int8 **v83; // r15
  char v84; // al
  unsigned __int8 *v85; // rax
  bool v86; // al
  __int64 v87; // rcx
  __int64 v88; // r8
  __int64 v89; // rsi
  _BYTE *v90; // rdi
  bool v91; // al
  unsigned __int64 v92; // r8
  unsigned __int64 v93; // rax
  bool v94; // al
  int v95; // r9d
  __int64 v96; // [rsp+8h] [rbp-E8h]
  unsigned int v97; // [rsp+14h] [rbp-DCh]
  __int64 v100; // [rsp+30h] [rbp-C0h]
  __int64 v101; // [rsp+30h] [rbp-C0h]
  __int64 v102; // [rsp+30h] [rbp-C0h]
  int v103; // [rsp+38h] [rbp-B8h]
  char v104; // [rsp+3Fh] [rbp-B1h]
  __int64 *v106; // [rsp+48h] [rbp-A8h]
  __int64 *v107; // [rsp+48h] [rbp-A8h]
  __int64 *v108; // [rsp+48h] [rbp-A8h]
  unsigned __int8 ****v109; // [rsp+48h] [rbp-A8h]
  __int64 v110; // [rsp+48h] [rbp-A8h]
  __int64 v111; // [rsp+48h] [rbp-A8h]
  __int64 v112; // [rsp+48h] [rbp-A8h]
  __int64 v113; // [rsp+48h] [rbp-A8h]
  __int64 v114; // [rsp+48h] [rbp-A8h]
  __int64 v115; // [rsp+48h] [rbp-A8h]
  unsigned __int64 v116; // [rsp+48h] [rbp-A8h]
  unsigned int v117; // [rsp+54h] [rbp-9Ch]
  char v118[32]; // [rsp+60h] [rbp-90h] BYREF
  __int16 v119; // [rsp+80h] [rbp-70h]
  _QWORD v120[4]; // [rsp+90h] [rbp-60h] BYREF
  __int16 v121; // [rsp+B0h] [rbp-40h]

  v7 = (char *)(a1 + 2);
  v104 = a5;
  v97 = (unsigned int)a4 >> 1;
  v103 = a4 & 1;
  *a1 = a1 + 2;
  a1[1] = 0x300000000LL;
  v8 = v103 + ((unsigned int)a4 >> 1);
  if ( v8 )
  {
    v9 = v103 + ((unsigned int)a4 >> 1);
    v10 = v8;
    i = a1 + 2;
    if ( v9 > 3 )
    {
      sub_C8D5F0((__int64)a1, v7, v10, 0x10u, a5, a6);
      v13 = *a1 + 16 * v10;
      for ( i = (_QWORD *)(*a1 + 16LL * *((unsigned int *)a1 + 2)); (_QWORD *)v13 != i; i += 2 )
      {
LABEL_4:
        if ( i )
        {
          *i = 0;
          i[1] = 0;
        }
      }
    }
    else
    {
      v12 = 16 * v10;
      v13 = (__int64)&v7[v12];
      if ( &v7[v12] != v7 )
        goto LABEL_4;
    }
    *((_DWORD *)a1 + 2) = v9;
  }
  if ( (a4 & 0xFFFFFFFE) != 0 )
  {
    v14 = a3;
    v117 = 0;
    do
    {
      v15 = v14[2];
      v16 = *a2;
      v17 = *(_QWORD *)(v15 + 48);
      v120[0] = v17;
      if ( v17 && (sub_B96E90((__int64)v120, v17, 1), (v19 = v120[0]) != 0) )
      {
        v20 = *(unsigned int *)(v16 + 8);
        v21 = *(unsigned __int64 **)v16;
        v22 = *(_DWORD *)(v16 + 8);
        v23 = (unsigned __int64 *)(*(_QWORD *)v16 + 16 * v20);
        if ( *(unsigned __int64 **)v16 != v23 )
        {
          while ( *(_DWORD *)v21 )
          {
            v21 += 2;
            if ( v23 == v21 )
              goto LABEL_50;
          }
          v21[1] = v120[0];
LABEL_17:
          sub_B91220((__int64)v120, v19);
          goto LABEL_18;
        }
LABEL_50:
        v62 = *(unsigned int *)(v16 + 12);
        if ( v20 >= v62 )
        {
          v92 = v20 + 1;
          v93 = v96 & 0xFFFFFFFF00000000LL;
          v96 &= 0xFFFFFFFF00000000LL;
          if ( v62 < v92 )
          {
            v116 = v93;
            sub_C8D5F0(v16, (const void *)(v16 + 16), v92, 0x10u, v92, v18);
            v93 = v116;
            v23 = (unsigned __int64 *)(*(_QWORD *)v16 + 16LL * *(unsigned int *)(v16 + 8));
          }
          *v23 = v93;
          v23[1] = v19;
          ++*(_DWORD *)(v16 + 8);
          v19 = v120[0];
        }
        else
        {
          if ( v23 )
          {
            *(_DWORD *)v23 = 0;
            v23[1] = v19;
            v22 = *(_DWORD *)(v16 + 8);
            v19 = v120[0];
          }
          *(_DWORD *)(v16 + 8) = v22 + 1;
        }
      }
      else
      {
        sub_93FB40(v16, 0);
        v19 = v120[0];
      }
      if ( v19 )
        goto LABEL_17;
LABEL_18:
      v24 = a2[1];
      v25 = (unsigned __int8 *)v14[1];
      v26 = *(_QWORD *)(v24 + 8);
      v27 = *(unsigned int *)(v24 + 24);
      if ( (_DWORD)v27 )
      {
        v28 = v27 - 1;
        v29 = (v27 - 1) & (((unsigned int)v25 >> 9) ^ ((unsigned int)v25 >> 4));
        v30 = (unsigned __int8 **)(v26 + 32LL * v29);
        v31 = *v30;
        if ( v25 == *v30 )
        {
LABEL_20:
          v32 = (unsigned __int8 **)(v26 + 32 * v27);
          if ( v30 != v32 )
            v25 = v30[3];
          v33 = (unsigned __int8 *)v14[3];
        }
        else
        {
          v65 = 1;
          while ( v31 != (unsigned __int8 *)-4096LL )
          {
            v95 = v65 + 1;
            v29 = v28 & (v65 + v29);
            v30 = (unsigned __int8 **)(v26 + 32LL * v29);
            v31 = *v30;
            if ( v25 == *v30 )
              goto LABEL_20;
            v65 = v95;
          }
          v33 = (unsigned __int8 *)v14[3];
          v32 = (unsigned __int8 **)(v26 + 32 * v27);
        }
        v34 = v28 & (((unsigned int)v33 >> 9) ^ ((unsigned int)v33 >> 4));
        v35 = (unsigned __int8 **)(v26 + 32LL * v34);
        v36 = *v35;
        if ( *v35 == v33 )
        {
LABEL_24:
          if ( v32 != v35 )
            v33 = v35[3];
        }
        else
        {
          v63 = 1;
          while ( v36 != (unsigned __int8 *)-4096LL )
          {
            v64 = v63 + 1;
            v34 = v28 & (v63 + v34);
            v35 = (unsigned __int8 **)(v26 + 32LL * v34);
            v36 = *v35;
            if ( *v35 == v33 )
              goto LABEL_24;
            v63 = v64;
          }
        }
      }
      else
      {
        v33 = (unsigned __int8 *)v14[3];
      }
      v37 = a2[2];
      if ( !**(_BYTE **)(v37 + 24) )
        goto LABEL_37;
      if ( *(_BYTE *)*v14 != 86 )
      {
LABEL_28:
        if ( *(_BYTE *)v15 != 86 )
          goto LABEL_29;
        goto LABEL_78;
      }
      v66 = *(_QWORD *)(*v14 + 8);
      if ( (unsigned int)*(unsigned __int8 *)(v66 + 8) - 17 <= 1 )
        v66 = **(_QWORD **)(v66 + 16);
      v100 = *v14;
      v110 = a2[2];
      v67 = sub_BCAC40(v66, 1);
      v37 = v110;
      v68 = v100;
      if ( !v67 )
        goto LABEL_99;
      if ( *(_BYTE *)v100 == 57 )
        goto LABEL_71;
      v69 = *(_QWORD *)(v100 + 8);
      if ( *(_BYTE *)v100 == 86 && *(_QWORD *)(*(_QWORD *)(v100 - 96) + 8LL) == v69 && **(_BYTE **)(v100 - 32) <= 0x15u )
      {
        v70 = sub_AC30F0(*(_QWORD *)(v100 - 32));
        v37 = v110;
        v68 = v100;
        if ( v70 )
          goto LABEL_71;
LABEL_99:
        v69 = *(_QWORD *)(v68 + 8);
      }
      if ( (unsigned int)*(unsigned __int8 *)(v69 + 8) - 17 <= 1 )
        v69 = **(_QWORD **)(v69 + 16);
      v115 = v37;
      v102 = v68;
      v86 = sub_BCAC40(v69, 1);
      v37 = v115;
      if ( !v86 )
        goto LABEL_28;
      v68 = v102;
      if ( *(_BYTE *)v102 != 58 )
      {
        if ( *(_BYTE *)v102 != 86 )
          goto LABEL_28;
        v89 = *(_QWORD *)(v102 + 8);
        if ( *(_QWORD *)(*(_QWORD *)(v102 - 96) + 8LL) != v89 )
          goto LABEL_28;
        v90 = *(_BYTE **)(v102 - 64);
        if ( *v90 > 0x15u )
          goto LABEL_28;
        v91 = sub_AD7A80(v90, v89, v102, v87, v88);
        v37 = v115;
        v68 = v102;
        if ( !v91 )
          goto LABEL_28;
      }
LABEL_71:
      if ( !v104 && *(unsigned __int8 **)(v37 + 16) == v25 )
        goto LABEL_37;
      v101 = v37;
      v111 = v68;
      sub_2B27770(v68);
      v71 = (*(_BYTE *)(v111 + 7) & 0x40) != 0
          ? *(unsigned __int8 ***)(v111 - 8)
          : (unsigned __int8 **)(v111 - 32LL * (*(_DWORD *)(v111 + 4) & 0x7FFFFFF));
      if ( *v71 == v25 )
        goto LABEL_37;
      v72 = sub_98ED70(v25, **(_QWORD **)(v101 + 32), 0, 0, 0);
      v37 = v101;
      if ( v72 )
        goto LABEL_37;
      if ( *(_BYTE *)v15 != 86 )
        goto LABEL_29;
LABEL_78:
      v73 = *(_QWORD *)(v15 + 8);
      if ( (unsigned int)*(unsigned __int8 *)(v73 + 8) - 17 <= 1 )
        v73 = **(_QWORD **)(v73 + 16);
      v112 = v37;
      v74 = sub_BCAC40(v73, 1);
      v37 = v112;
      if ( !v74 )
      {
LABEL_118:
        v75 = *(_QWORD *)(v15 + 8);
LABEL_84:
        if ( (unsigned int)*(unsigned __int8 *)(v75 + 8) - 17 <= 1 )
          v75 = **(_QWORD **)(v75 + 16);
        v113 = v37;
        v76 = sub_BCAC40(v75, 1);
        v37 = v113;
        if ( !v76
          || *(_BYTE *)v15 != 58
          && (*(_BYTE *)v15 != 86
           || (v80 = *(_QWORD *)(v15 + 8), *(_QWORD *)(*(_QWORD *)(v15 - 96) + 8LL) != v80)
           || (v81 = *(_BYTE **)(v15 - 64), *v81 > 0x15u)
           || (v82 = sub_AD7A80(v81, v80, v77, v78, v79), v37 = v113, !v82)) )
        {
LABEL_29:
          if ( *(unsigned __int8 **)(v37 + 16) != v25 )
          {
            v38 = *(__int64 **)(v37 + 40);
            v119 = 257;
            v106 = v38;
            v121 = 257;
            v39 = sub_BD2C40(72, 1u);
            v40 = v106;
            v41 = (__int64)v39;
            if ( v39 )
            {
              sub_B549F0((__int64)v39, (__int64)v25, (__int64)v120, 0, 0);
              v40 = v106;
            }
            v107 = v40;
            (*(void (__fastcall **)(__int64, __int64, char *, __int64, __int64))(*(_QWORD *)v40[11] + 16LL))(
              v40[11],
              v41,
              v118,
              v40[7],
              v40[8]);
            v42 = *v107;
            v43 = *v107 + 16LL * *((unsigned int *)v107 + 2);
            if ( *v107 != v43 )
            {
              v108 = v14;
              v44 = v43;
              v45 = v42;
              do
              {
                v46 = *(_QWORD *)(v45 + 8);
                v47 = *(_DWORD *)v45;
                v45 += 16;
                sub_B99FD0(v41, v47, v46);
              }
              while ( v44 != v45 );
              v14 = v108;
            }
            v25 = (unsigned __int8 *)v41;
          }
          goto LABEL_37;
        }
        goto LABEL_92;
      }
      if ( *(_BYTE *)v15 != 57 )
      {
        v75 = *(_QWORD *)(v15 + 8);
        if ( *(_BYTE *)v15 != 86 || *(_QWORD *)(*(_QWORD *)(v15 - 96) + 8LL) != v75 || **(_BYTE **)(v15 - 32) > 0x15u )
          goto LABEL_84;
        v94 = sub_AC30F0(*(_QWORD *)(v15 - 32));
        v37 = v112;
        if ( !v94 )
          goto LABEL_118;
      }
LABEL_92:
      if ( v104 || *(unsigned __int8 **)(v37 + 16) != v33 )
      {
        v114 = v37;
        sub_2B27770(v15);
        v83 = (*(_BYTE *)(v15 + 7) & 0x40) != 0
            ? *(unsigned __int8 ***)(v15 - 8)
            : (unsigned __int8 **)(v15 - 32LL * (*(_DWORD *)(v15 + 4) & 0x7FFFFFF));
        if ( *v83 != v33 )
        {
          v84 = sub_98ED70(v33, **(_QWORD **)(v114 + 32), 0, 0, 0);
          v37 = v114;
          if ( !v84 )
            goto LABEL_29;
        }
      }
      v85 = v25;
      v25 = v33;
      v33 = v85;
LABEL_37:
      v48 = (unsigned __int8 ****)a2[3];
      v49 = *a2;
      v120[0] = "op.rdx";
      v50 = 1;
      v121 = 259;
      v51 = *((unsigned int *)v48 + 2);
      v52 = *((_DWORD *)v48 + 394);
      if ( v51 != 2 )
      {
        v50 = 0;
        if ( v51 == 1 )
        {
          v53 = sub_2B0AAE0(**v48, (__int64)&(**v48)[*((unsigned int *)*v48 + 2)]);
          v50 = v54 != v53;
        }
      }
      v109 = v48;
      v55 = (unsigned __int8 *)sub_2B21610(v49, v52, (__int64)v25, (__int64)v33, (__int64)v120, v50);
      v56 = v55;
      if ( v52 - 6 <= 3 && *v55 == 86 )
      {
        sub_F70480(*((unsigned __int8 **)v55 - 12), **v109, *((unsigned int *)*v109 + 2), 0, 0);
        sub_F70480(v56, (*v109)[18], *((unsigned int *)*v109 + 38), 0, 0);
      }
      else
      {
        sub_F70480(v55, **v109, *((unsigned int *)*v109 + 2), 0, 0);
      }
      v14 += 4;
      v57 = v117 >> 1;
      v117 += 2;
      v58 = (_QWORD *)(*a1 + 16 * v57);
      *v58 = *(v14 - 4);
      v58[1] = v56;
    }
    while ( ((unsigned int)a4 & 0xFFFFFFFE) > v117 );
  }
  if ( v103 )
  {
    v59 = &a3[2 * a4 - 2];
    v60 = (__int64 *)(*a1 + 16LL * v97);
    *v60 = *v59;
    v60[1] = v59[1];
  }
  return a1;
}
