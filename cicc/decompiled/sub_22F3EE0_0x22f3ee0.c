// Function: sub_22F3EE0
// Address: 0x22f3ee0
//
__int64 *__fastcall sub_22F3EE0(
        __int64 *a1,
        __int64 *a2,
        __int64 (__fastcall ***a3)(_QWORD),
        int a4,
        char *a5,
        unsigned int *a6)
{
  const char *v10; // rax
  size_t v11; // r11
  unsigned __int8 *v12; // rsi
  unsigned int v13; // edx
  __int64 v14; // rax
  __int64 v15; // r15
  unsigned __int8 *v16; // r12
  __int64 v17; // rax
  __int64 v18; // rbx
  __int64 v20; // rax
  __int64 v21; // rdx
  char *v22; // r11
  __int64 v23; // r12
  __int64 v24; // r8
  __int64 v25; // r9
  char *v26; // r14
  __int64 v27; // rax
  unsigned int v28; // r14d
  __int64 (__fastcall *v29)(_QWORD); // rax
  __int64 v30; // r8
  __int64 v31; // r9
  __int64 v32; // r14
  __int64 v33; // rax
  __int64 v34; // rax
  unsigned int v35; // r9d
  __int64 v36; // r15
  unsigned __int8 *v37; // rbx
  __int64 v38; // rax
  char *v39; // r10
  __int64 v40; // r12
  char *v41; // rbx
  char v42; // r15
  char *v43; // rax
  void *v44; // r14
  __int64 v45; // r8
  __int64 v46; // r9
  __int64 v47; // rax
  unsigned __int64 v48; // rcx
  unsigned int v49; // edx
  __int64 v50; // rax
  __int64 v51; // r15
  unsigned __int8 *v52; // r12
  __int64 (__fastcall **v53)(_QWORD); // rax
  __int64 v54; // rsi
  __int64 v55; // rax
  unsigned int v56; // r9d
  __int64 v57; // r15
  unsigned __int8 *v58; // r12
  __int64 v59; // rsi
  __int64 v60; // rax
  unsigned int v61; // r14d
  __int64 (__fastcall *v62)(_QWORD); // rax
  __int64 v63; // r8
  __int64 v64; // r9
  __int64 v65; // r14
  __int64 v66; // rax
  unsigned int v67; // edx
  __int64 v68; // rax
  __int64 v69; // rdx
  __int64 v70; // r8
  __int64 v71; // r9
  _DWORD *v72; // r11
  int v73; // ecx
  int v74; // r14d
  unsigned __int64 v75; // rcx
  int v76; // eax
  __int64 v77; // rax
  __int64 v78; // rax
  unsigned __int8 *v79; // rax
  unsigned int v80; // esi
  __int64 v81; // r15
  unsigned __int8 *v82; // r12
  __int64 v83; // rax
  __int64 v84; // [rsp-10h] [rbp-70h]
  __int64 v85; // [rsp-8h] [rbp-68h]
  __int64 v86; // [rsp+8h] [rbp-58h]
  __int64 v87; // [rsp+10h] [rbp-50h]
  unsigned int v88; // [rsp+10h] [rbp-50h]
  unsigned int v89; // [rsp+10h] [rbp-50h]
  unsigned __int8 *v90; // [rsp+10h] [rbp-50h]
  size_t v91; // [rsp+10h] [rbp-50h]
  int v92; // [rsp+18h] [rbp-48h]
  __int64 v93; // [rsp+18h] [rbp-48h]
  unsigned int v94; // [rsp+18h] [rbp-48h]
  __int64 v95; // [rsp+18h] [rbp-48h]
  unsigned int v96; // [rsp+18h] [rbp-48h]
  __int64 v97; // [rsp+18h] [rbp-48h]
  int v98; // [rsp+18h] [rbp-48h]
  const void *v99; // [rsp+18h] [rbp-48h]
  char *src; // [rsp+20h] [rbp-40h]
  char *srca; // [rsp+20h] [rbp-40h]
  char *srcb; // [rsp+20h] [rbp-40h]
  char *srci; // [rsp+20h] [rbp-40h]
  int srcc; // [rsp+20h] [rbp-40h]
  char *srcd; // [rsp+20h] [rbp-40h]
  unsigned __int8 *srce; // [rsp+20h] [rbp-40h]
  void *srcf; // [rsp+20h] [rbp-40h]
  void *srcg; // [rsp+20h] [rbp-40h]
  unsigned int srch; // [rsp+20h] [rbp-40h]
  size_t nb; // [rsp+28h] [rbp-38h]
  size_t nc; // [rsp+28h] [rbp-38h]
  size_t na; // [rsp+28h] [rbp-38h]
  size_t nd; // [rsp+28h] [rbp-38h]

  v10 = (const char *)((__int64 (__fastcall *)(__int64 (__fastcall ***)(_QWORD), _QWORD))**a3)(a3, *a6);
  v11 = 0;
  if ( v10 )
    v11 = strlen(v10);
  v12 = (unsigned __int8 *)*a2;
  switch ( *(_BYTE *)(*a2 + 44) )
  {
    case 3:
      if ( a5 != (char *)v11 )
        goto LABEL_9;
      srch = (*a6)++;
      v81 = a2[1];
      v82 = (unsigned __int8 *)*a2;
      v83 = sub_22077B0(0x58u);
      v18 = v83;
      if ( v83 )
        sub_314D310(v83, (_DWORD)v82, v81, a4, (_DWORD)a5, srch, 0);
      goto LABEL_43;
    case 4:
      v53 = *a3;
      v54 = *a6;
      goto LABEL_41;
    case 6:
      if ( a5 == (char *)v11 )
      {
        v49 = *a6 + 2;
        *a6 = v49;
        if ( v49 <= (unsigned int)(*a3)[1](a3) )
          goto LABEL_37;
      }
      goto LABEL_9;
    case 7:
      if ( a5 != (char *)v11 )
        goto LABEL_9;
      v89 = (*a6)++;
      v59 = *a2;
      v97 = a2[1];
      srce = (unsigned __int8 *)*a2;
      v60 = sub_22077B0(0x58u);
      v23 = v60;
      if ( v60 )
      {
        sub_314D310(v60, (_DWORD)srce, v97, a4, (_DWORD)a5, v89, 0);
        v59 = v85;
      }
      while ( 1 )
      {
        v61 = *a6;
        if ( v61 >= ((unsigned int (__fastcall *)(__int64 (__fastcall ***)(_QWORD), __int64))(*a3)[1])(a3, v59)
          || !((__int64 (__fastcall *)(__int64 (__fastcall ***)(_QWORD), _QWORD))**a3)(a3, *a6) )
        {
          break;
        }
        v59 = *a6;
        v62 = **a3;
        *a6 = v59 + 1;
        v65 = v62(a3);
        v66 = *(unsigned int *)(v23 + 56);
        if ( v66 + 1 > (unsigned __int64)*(unsigned int *)(v23 + 60) )
        {
          v59 = v23 + 64;
          sub_C8D5F0(v23 + 48, (const void *)(v23 + 64), v66 + 1, 8u, v63, v64);
          v66 = *(unsigned int *)(v23 + 56);
        }
        *(_QWORD *)(*(_QWORD *)(v23 + 48) + 8 * v66) = v65;
        ++*(_DWORD *)(v23 + 56);
      }
LABEL_64:
      *a1 = v23;
      return a1;
    case 8:
      srca = (char *)v11;
      v88 = *a6;
      v93 = a2[1];
      v20 = sub_22077B0(0x58u);
      v22 = srca;
      v23 = v20;
      if ( v20 )
      {
        sub_314D310(v20, (_DWORD)v12, v93, a4, (_DWORD)a5, v88, 0);
        v22 = srca;
        v21 = v85;
      }
      if ( a5 != v22 )
      {
        v12 = (unsigned __int8 *)*a6;
        v26 = &a5[((__int64 (__fastcall *)(__int64 (__fastcall ***)(_QWORD), unsigned __int8 *, __int64))**a3)(
                    a3,
                    v12,
                    v21)];
        v27 = *(unsigned int *)(v23 + 56);
        if ( v27 + 1 > (unsigned __int64)*(unsigned int *)(v23 + 60) )
        {
          v12 = (unsigned __int8 *)(v23 + 64);
          sub_C8D5F0(v23 + 48, (const void *)(v23 + 64), v27 + 1, 8u, v24, v25);
          v27 = *(unsigned int *)(v23 + 56);
        }
        v21 = *(_QWORD *)(v23 + 48);
        *(_QWORD *)(v21 + 8 * v27) = v26;
        ++*(_DWORD *)(v23 + 56);
      }
      v28 = *a6 + 1;
      *a6 = v28;
      while ( ((unsigned int (__fastcall *)(__int64 (__fastcall ***)(_QWORD), unsigned __int8 *, __int64))(*a3)[1])(
                a3,
                v12,
                v21) > v28
           && ((__int64 (__fastcall *)(__int64 (__fastcall ***)(_QWORD), _QWORD))**a3)(a3, *a6) )
      {
        v12 = (unsigned __int8 *)*a6;
        v29 = **a3;
        *a6 = (_DWORD)v12 + 1;
        v32 = v29(a3);
        v33 = *(unsigned int *)(v23 + 56);
        if ( v33 + 1 > (unsigned __int64)*(unsigned int *)(v23 + 60) )
        {
          v12 = (unsigned __int8 *)(v23 + 64);
          sub_C8D5F0(v23 + 48, (const void *)(v23 + 64), v33 + 1, 8u, v30, v31);
          v33 = *(unsigned int *)(v23 + 56);
        }
        v21 = *(_QWORD *)(v23 + 48);
        *(_QWORD *)(v21 + 8 * v33) = v32;
        v28 = *a6;
        ++*(_DWORD *)(v23 + 56);
      }
      goto LABEL_64;
    case 9:
      v34 = ((__int64 (__fastcall *)(__int64 (__fastcall ***)(_QWORD), _QWORD))**a3)(a3, *a6);
      v35 = *a6;
      srcb = &a5[v34];
      ++*a6;
      v36 = a2[1];
      v37 = (unsigned __int8 *)*a2;
      v94 = v35;
      v38 = sub_22077B0(0x58u);
      v39 = srcb;
      v40 = v38;
      if ( v38 )
      {
        sub_314D310(v38, (_DWORD)v37, v36, a4, (_DWORD)a5, v94, 0);
        v39 = srcb;
      }
      v41 = v39;
      while ( 2 )
      {
        v42 = *v41;
        if ( *v41 )
        {
          v43 = v41 + 1;
          if ( v42 != 44 )
            goto LABEL_26;
        }
        if ( v41 != v39 )
        {
          srci = v39;
          nb = v41 - v39;
          v44 = (void *)sub_2207820(v41 - v39 + 1);
          memcpy(v44, srci, nb);
          v47 = *(unsigned int *)(v40 + 56);
          v48 = *(unsigned int *)(v40 + 60);
          *((_BYTE *)v44 + nb) = 0;
          if ( v47 + 1 > v48 )
          {
            sub_C8D5F0(v40 + 48, (const void *)(v40 + 64), v47 + 1, 8u, v45, v46);
            v47 = *(unsigned int *)(v40 + 56);
          }
          *(_QWORD *)(*(_QWORD *)(v40 + 48) + 8 * v47) = v44;
          ++*(_DWORD *)(v40 + 56);
        }
        if ( v42 )
        {
          v43 = v41 + 1;
          v39 = v41 + 1;
LABEL_26:
          v41 = v43;
          continue;
        }
        break;
      }
      *(_BYTE *)(v40 + 44) |= 4u;
      *a1 = v40;
      return a1;
    case 0xA:
      if ( a5 != (char *)v11 )
        goto LABEL_9;
      v67 = *a6 + v12[45] + 1;
      *a6 = v67;
      if ( v67 > (unsigned int)(*a3)[1](a3) )
        goto LABEL_9;
      v86 = ((__int64 (__fastcall *)(__int64 (__fastcall ***)(_QWORD), _QWORD))**a3)(
              a3,
              *a6 - *(unsigned __int8 *)(*a2 + 45));
      v90 = (unsigned __int8 *)*a2;
      srcf = (void *)a2[1];
      v98 = *a6 - 1 - *(unsigned __int8 *)(*a2 + 45);
      v68 = sub_22077B0(0x58u);
      v72 = (_DWORD *)v68;
      if ( v68 )
      {
        v73 = a4;
        nc = v68;
        sub_314D360(v68, (_DWORD)v90, (_DWORD)srcf, v73, (_DWORD)a5, v98, v86, 0);
        v72 = (_DWORD *)nc;
        v71 = v84;
      }
      v74 = 1;
      srcg = v72 + 12;
      v75 = (unsigned __int64)(v72 + 16);
      v76 = *(unsigned __int8 *)(*a2 + 45);
      v99 = v72 + 16;
      if ( v76 != 1 )
      {
        do
        {
          na = (size_t)v72;
          v77 = ((__int64 (__fastcall *)(__int64 (__fastcall ***)(_QWORD), _QWORD, __int64, unsigned __int64, __int64, __int64))**a3)(
                  a3,
                  v74 + *a6 - v76,
                  v69,
                  v75,
                  v70,
                  v71);
          v72 = (_DWORD *)na;
          v71 = v77;
          v78 = *(unsigned int *)(na + 56);
          v75 = *(unsigned int *)(na + 60);
          if ( v78 + 1 > v75 )
          {
            v91 = na;
            nd = v71;
            sub_C8D5F0((__int64)srcg, v99, v78 + 1, 8u, v70, v71);
            v72 = (_DWORD *)v91;
            v71 = nd;
            v78 = *(unsigned int *)(v91 + 56);
          }
          v69 = *((_QWORD *)v72 + 6);
          ++v74;
          *(_QWORD *)(v69 + 8 * v78) = v71;
          v79 = (unsigned __int8 *)*a2;
          ++v72[14];
          v76 = v79[45];
        }
        while ( v74 != v76 );
      }
      *a1 = (__int64)v72;
      return a1;
    case 0xB:
      v54 = *a6;
      if ( a5 != (char *)v11 )
      {
        v53 = *a3;
LABEL_41:
        v55 = ((__int64 (__fastcall *)(__int64 (__fastcall ***)(_QWORD), __int64))*v53)(a3, v54);
        v56 = *a6;
        srcd = &a5[v55];
        ++*a6;
        v57 = a2[1];
        v96 = v56;
        v58 = (unsigned __int8 *)*a2;
        v18 = sub_22077B0(0x58u);
        if ( v18 )
          sub_314D360(v18, (_DWORD)v58, v57, a4, (_DWORD)a5, v96, (__int64)srcd, 0);
        goto LABEL_43;
      }
      v80 = v54 + 2;
      *a6 = v80;
      if ( v80 > (unsigned int)(*a3)[1](a3) )
        goto LABEL_9;
LABEL_37:
      if ( !((__int64 (__fastcall *)(__int64 (__fastcall ***)(_QWORD), _QWORD))**a3)(a3, *a6 - 1) )
      {
LABEL_9:
        *a1 = 0;
        return a1;
      }
      v50 = ((__int64 (__fastcall *)(__int64 (__fastcall ***)(_QWORD), _QWORD))**a3)(a3, *a6 - 1);
      v51 = a2[1];
      v95 = v50;
      v52 = (unsigned __int8 *)*a2;
      srcc = *a6 - 2;
      v18 = sub_22077B0(0x58u);
      if ( v18 )
        sub_314D360(v18, (_DWORD)v52, v51, a4, (_DWORD)a5, srcc, v95, 0);
LABEL_43:
      *a1 = v18;
      return a1;
    case 0xC:
      v13 = *a6 + 2;
      *a6 = v13;
      if ( v13 > (unsigned int)(*a3)[1](a3)
        || !((__int64 (__fastcall *)(__int64 (__fastcall ***)(_QWORD), _QWORD))**a3)(a3, *a6 - 1) )
      {
        goto LABEL_9;
      }
      v87 = ((__int64 (__fastcall *)(__int64 (__fastcall ***)(_QWORD), _QWORD))**a3)(a3, *a6 - 1);
      v14 = ((__int64 (__fastcall *)(__int64 (__fastcall ***)(_QWORD), _QWORD))**a3)(a3, *a6 - 2);
      v15 = a2[1];
      v16 = (unsigned __int8 *)*a2;
      src = &a5[v14];
      v92 = *a6 - 2;
      v17 = sub_22077B0(0x58u);
      v18 = v17;
      if ( v17 )
        sub_314D3B0(v17, (_DWORD)v16, v15, a4, (_DWORD)a5, v92, (__int64)src, v87, 0);
      goto LABEL_43;
    default:
      BUG();
  }
}
