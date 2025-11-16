// Function: sub_2053040
// Address: 0x2053040
//
void __fastcall sub_2053040(
        __int64 a1,
        int *a2,
        __m128 a3,
        double a4,
        __m128i a5,
        __int64 a6,
        __int64 a7,
        __int64 a8,
        int a9)
{
  int *v10; // rbx
  __int64 *v11; // r12
  __int64 v12; // rdx
  __int64 v13; // rax
  int v14; // r13d
  __int64 *v15; // rax
  int v16; // edx
  int v17; // edx
  __int64 v18; // r8
  __int64 v19; // rdx
  __int64 v20; // r9
  __int64 *v21; // rax
  __int64 v22; // rsi
  int v23; // edx
  __int64 *v24; // r13
  __int64 v25; // rdx
  __int64 v26; // r12
  int v27; // eax
  __int64 *v28; // rdi
  __int64 v29; // r8
  __int64 v30; // rdx
  __int64 v31; // r12
  __int64 *v32; // rax
  __int64 *v33; // rsi
  __int64 v34; // r9
  __int64 v35; // rax
  size_t v36; // r12
  __int64 v37; // rax
  __int64 *v38; // r9
  __int64 *v39; // rax
  __int64 v40; // rax
  __int64 v41; // r13
  size_t v42; // rbx
  void *v43; // rdi
  __int64 v44; // rax
  int v45; // edx
  __int64 v46; // r13
  __int64 v47; // rbx
  int v48; // r14d
  __int64 v49; // rax
  __int64 v50; // rsi
  unsigned int v51; // eax
  __int64 v52; // rcx
  __int64 v53; // r8
  int v54; // r9d
  __int64 v55; // rax
  __int64 v56; // rdx
  __int64 *v57; // r12
  int v58; // r11d
  __int64 *v59; // rax
  int v60; // edx
  unsigned int v61; // r13d
  int v62; // edx
  __int64 v63; // r8
  __int64 v64; // r9
  __int64 v65; // rdx
  __int64 v66; // r9
  __int64 v67; // rdx
  __int64 v68; // r13
  unsigned __int8 v69; // r9
  _QWORD *v70; // rax
  __int64 v71; // rdx
  __int64 *v72; // r14
  _QWORD *v73; // r8
  __int64 v74; // r9
  unsigned __int64 v75; // rcx
  __int64 v76; // rax
  __int64 v77; // rsi
  __int64 *v78; // rbx
  int v79; // edx
  int v80; // r13d
  __int64 v81; // r12
  int v82; // eax
  __int64 v83; // rdx
  __int128 v84; // [rsp-20h] [rbp-130h]
  __int128 v85; // [rsp-10h] [rbp-120h]
  __int128 v86; // [rsp-10h] [rbp-120h]
  __int64 v87; // [rsp-8h] [rbp-118h]
  int v88; // [rsp+4h] [rbp-10Ch]
  __int64 v89; // [rsp+8h] [rbp-108h]
  unsigned __int8 v90; // [rsp+8h] [rbp-108h]
  const void ***srcb; // [rsp+10h] [rbp-100h]
  void *srcc; // [rsp+10h] [rbp-100h]
  void *srcd; // [rsp+10h] [rbp-100h]
  __int64 src; // [rsp+10h] [rbp-100h]
  __int64 *srce; // [rsp+10h] [rbp-100h]
  __int64 *srca; // [rsp+10h] [rbp-100h]
  __int64 *srcf; // [rsp+10h] [rbp-100h]
  unsigned __int8 srcg; // [rsp+10h] [rbp-100h]
  int srch; // [rsp+10h] [rbp-100h]
  _QWORD *srci; // [rsp+10h] [rbp-100h]
  __int64 v101; // [rsp+18h] [rbp-F8h]
  int destb; // [rsp+20h] [rbp-F0h]
  __int64 *dest; // [rsp+20h] [rbp-F0h]
  int destc; // [rsp+20h] [rbp-F0h]
  int destd; // [rsp+20h] [rbp-F0h]
  __int64 *desta; // [rsp+20h] [rbp-F0h]
  int v107; // [rsp+30h] [rbp-E0h]
  int v108; // [rsp+30h] [rbp-E0h]
  const void ***v109; // [rsp+30h] [rbp-E0h]
  __int64 v110; // [rsp+38h] [rbp-D8h]
  unsigned __int8 v111; // [rsp+38h] [rbp-D8h]
  __int64 v112; // [rsp+38h] [rbp-D8h]
  __int64 v113; // [rsp+80h] [rbp-90h] BYREF
  int v114; // [rsp+88h] [rbp-88h]
  __int64 v115; // [rsp+90h] [rbp-80h] BYREF
  int v116; // [rsp+98h] [rbp-78h]
  _QWORD *v117; // [rsp+A0h] [rbp-70h]
  __int64 v118; // [rsp+A8h] [rbp-68h]
  __int64 v119; // [rsp+B0h] [rbp-60h]
  __int64 v120; // [rsp+B8h] [rbp-58h]
  size_t v121; // [rsp+C0h] [rbp-50h]
  __int64 v122; // [rsp+C8h] [rbp-48h] BYREF
  int v123; // [rsp+D0h] [rbp-40h]

  v10 = a2;
  v11 = *(__int64 **)(a1 + 552);
  if ( (unsigned int)(*(_DWORD *)(*(_QWORD *)(a1 + 544) + 504LL) - 34) <= 1 )
  {
    v12 = *(unsigned int *)(a1 + 536);
    v13 = *(_QWORD *)a1;
    v113 = 0;
    v14 = *a2;
    v114 = v12;
    if ( v13 )
    {
      v12 = v13 + 48;
      if ( &v113 != (__int64 *)(v13 + 48) )
      {
        a2 = *(int **)(v13 + 48);
        v113 = (__int64)a2;
        if ( a2 )
          sub_1623A60((__int64)&v113, (__int64)a2, 2);
      }
    }
    v15 = sub_2051DF0((__int64 *)a1, *(double *)a3.m128_u64, a4, a5, (__int64)a2, v12, a7, a8, a9);
    v107 = v16;
    v110 = (__int64)v15;
    srcb = (const void ***)sub_1D252B0((__int64)v11, 5, 0, 1, 0);
    destb = v17;
    v116 = v107;
    v115 = v110;
    v117 = sub_1D2A660(v11, v14, 5u, 0, v18, 5);
    v118 = v19;
    *((_QWORD *)&v85 + 1) = 2;
    *(_QWORD *)&v85 = &v115;
    v21 = sub_1D36D80(v11, 47, (__int64)&v113, srcb, destb, *(double *)a3.m128_u64, a4, a5, v20, v85);
    v22 = v113;
    v108 = v23;
    v24 = v21;
    v25 = v87;
    if ( v113 )
      sub_161E7C0((__int64)&v113, v113);
    v26 = *(_QWORD *)(*(_QWORD *)(a1 + 712) + 8LL);
    v27 = (*(__int64 (__fastcall **)(_QWORD, __int64, __int64))(**(_QWORD **)(*(_QWORD *)(a1 + 552) + 16LL) + 1016LL))(
            *(_QWORD *)(*(_QWORD *)(a1 + 552) + 16LL),
            v22,
            v25);
    v28 = (__int64 *)v26;
    v29 = sub_1E0BA80(v26, v27);
    v30 = 3LL * (unsigned int)v10[1];
    v31 = *(_QWORD *)(v29 + 8) + 24LL * (unsigned int)v10[1];
    v32 = *(__int64 **)(v31 + 8);
    v33 = *(__int64 **)v31;
    v34 = (__int64)v32 - *(_QWORD *)v31;
    v89 = v34;
    if ( v34 )
    {
      if ( (unsigned __int64)v34 > 0x7FFFFFFFFFFFFFF8LL )
        goto LABEL_70;
      v28 = (__int64 *)(*(_QWORD *)(v31 + 8) - *(_QWORD *)v31);
      v35 = sub_22077B0(v34);
      v33 = *(__int64 **)v31;
      dest = (__int64 *)v35;
      v32 = *(__int64 **)(v31 + 8);
      v34 = (__int64)v32 - *(_QWORD *)v31;
      v36 = v34;
    }
    else
    {
      dest = 0;
      v36 = 0;
    }
    if ( v33 != v32 )
    {
      v28 = dest;
      srcc = (void *)v34;
      memmove(dest, v33, v36);
      v34 = (__int64)srcc;
    }
    v30 = *(unsigned int *)(a1 + 536);
    v37 = *(_QWORD *)a1;
    v113 = 0;
    v114 = v30;
    if ( v37 )
    {
      v30 = v37 + 48;
      if ( &v113 != (__int64 *)(v37 + 48) )
      {
        v33 = *(__int64 **)(v37 + 48);
        v113 = (__int64)v33;
        if ( v33 )
        {
          v28 = &v113;
          srcd = (void *)v34;
          sub_1623A60((__int64)&v113, (__int64)v33, 2);
          v34 = (__int64)srcd;
        }
      }
    }
    src = v34 >> 3;
    if ( v34 >> 3 )
    {
      if ( v36 > 0x7FFFFFFFFFFFFFF8LL )
        goto LABEL_70;
      v38 = (__int64 *)sub_22077B0(v36);
    }
    else
    {
      v38 = 0;
    }
    if ( !v36 )
    {
      v82 = v10[1];
      v83 = *(_QWORD *)(a1 + 552);
      v117 = v24;
      v119 = 0;
      v116 = v82;
      v115 = v83;
      v41 = v113;
      LODWORD(v118) = v108;
      v120 = 0;
      v121 = 0;
      if ( !src )
        goto LABEL_60;
      goto LABEL_23;
    }
    v33 = dest;
    v28 = v38;
    v39 = (__int64 *)memcpy(v38, dest, v36);
    v30 = *(_QWORD *)(a1 + 552);
    v117 = v24;
    v38 = v39;
    LODWORD(v39) = v10[1];
    v119 = 0;
    v115 = v30;
    v116 = (int)v39;
    v120 = 0;
    LODWORD(v118) = v108;
    v121 = 0;
    if ( !src )
    {
      v121 = v36;
      v41 = v113;
      v42 = v36;
      v43 = 0;
LABEL_24:
      srca = v38;
      memcpy(v43, v38, v36);
      v120 = v42;
      v38 = srca;
      v122 = v41;
      if ( !v41 )
      {
        v123 = v114;
        j_j___libc_free_0(srca, v36);
        goto LABEL_26;
      }
      goto LABEL_25;
    }
    if ( v36 <= 0x7FFFFFFFFFFFFFF8LL )
    {
LABEL_23:
      srce = v38;
      v40 = sub_22077B0(v36);
      v41 = v113;
      v38 = srce;
      v42 = v40 + v36;
      v119 = v40;
      v43 = (void *)v40;
      v120 = v40;
      v121 = v40 + v36;
      if ( v36 )
        goto LABEL_24;
LABEL_60:
      v122 = v41;
      if ( !v41 )
      {
        v123 = v114;
        if ( !v38 )
          goto LABEL_26;
        goto LABEL_62;
      }
LABEL_25:
      srcf = v38;
      sub_1623A60((__int64)&v122, v41, 2);
      v38 = srcf;
      v123 = v114;
      if ( !srcf )
      {
LABEL_26:
        if ( v113 )
          sub_161E7C0((__int64)&v113, v113);
        v44 = sub_217AEF0(&v115);
        v46 = *(_QWORD *)(a1 + 552);
        v47 = v44;
        v48 = v45;
        if ( v44 )
        {
          nullsub_686();
          *(_QWORD *)(v46 + 176) = v47;
          *(_DWORD *)(v46 + 184) = v48;
          sub_1D23870();
        }
        else
        {
          *(_QWORD *)(v46 + 176) = 0;
          *(_DWORD *)(v46 + 184) = v45;
        }
        if ( v122 )
          sub_161E7C0((__int64)&v122, v122);
        if ( v119 )
          j_j___libc_free_0(v119, v121 - v119);
        if ( dest )
          j_j___libc_free_0(dest, v89);
        return;
      }
LABEL_62:
      j_j___libc_free_0(v38, v36);
      goto LABEL_26;
    }
LABEL_70:
    sub_4261EA(v28, v33, v30);
  }
  v49 = sub_1E0A0C0(v11[4]);
  v50 = 0;
  v51 = 8 * sub_15A9520(v49, 0);
  if ( v51 == 32 )
  {
    v54 = 5;
  }
  else if ( v51 > 0x20 )
  {
    v54 = 6;
    if ( v51 != 64 )
    {
      v54 = 0;
      if ( v51 == 128 )
        v54 = 7;
    }
  }
  else
  {
    v54 = 3;
    if ( v51 != 8 )
    {
      LOBYTE(v54) = v51 == 16;
      v54 *= 4;
    }
  }
  v55 = *(_QWORD *)a1;
  v56 = *(unsigned int *)(a1 + 536);
  v113 = 0;
  v57 = *(__int64 **)(a1 + 552);
  v58 = *v10;
  v114 = v56;
  if ( v55 )
  {
    v56 = v55 + 48;
    if ( &v113 != (__int64 *)(v55 + 48) )
    {
      v50 = *(_QWORD *)(v55 + 48);
      v113 = v50;
      if ( v50 )
      {
        destc = v58;
        v111 = v54;
        sub_1623A60((__int64)&v113, v50, 2);
        v58 = destc;
        v54 = v111;
      }
    }
  }
  v88 = v58;
  srcg = v54;
  v59 = sub_2051DF0((__int64 *)a1, *(double *)a3.m128_u64, a4, a5, v50, v56, v52, v53, v54);
  destd = v60;
  v112 = (__int64)v59;
  v61 = srcg;
  v90 = srcg;
  v109 = (const void ***)sub_1D252B0((__int64)v57, srcg, 0, 1, 0);
  srch = v62;
  v115 = v112;
  v116 = destd;
  v117 = sub_1D2A660(v57, v88, v61, 0, v63, v64);
  v118 = v65;
  *((_QWORD *)&v86 + 1) = 2;
  *(_QWORD *)&v86 = &v115;
  desta = sub_1D36D80(v57, 47, (__int64)&v113, v109, srch, *(double *)a3.m128_u64, a4, a5, v66, v86);
  v68 = v67;
  v69 = v90;
  if ( v113 )
  {
    sub_161E7C0((__int64)&v113, v113);
    v69 = v90;
  }
  v70 = sub_1D29EE0(*(_QWORD **)(a1 + 552), v10[1], v69, 0, 0, 0);
  v115 = 0;
  v72 = *(__int64 **)(a1 + 552);
  v73 = v70;
  v74 = v71;
  v75 = (unsigned __int64)desta;
  v76 = *(_QWORD *)a1;
  v116 = *(_DWORD *)(a1 + 536);
  if ( v76 )
  {
    if ( &v115 != (__int64 *)(v76 + 48) )
    {
      v77 = *(_QWORD *)(v76 + 48);
      v115 = v77;
      if ( v77 )
      {
        srci = v73;
        v101 = v71;
        sub_1623A60((__int64)&v115, v77, 2);
        v73 = srci;
        v74 = v101;
        v75 = (unsigned __int64)desta;
      }
    }
  }
  *((_QWORD *)&v84 + 1) = v74;
  *(_QWORD *)&v84 = v73;
  v78 = sub_1D3A900(v72, 0xBEu, (__int64)&v115, 1u, 0, 0, a3, a4, a5, v75, (__int16 *)1, v84, (__int64)desta, v68);
  v80 = v79;
  if ( v115 )
    sub_161E7C0((__int64)&v115, v115);
  v81 = *(_QWORD *)(a1 + 552);
  if ( v78 )
  {
    nullsub_686();
    *(_QWORD *)(v81 + 176) = v78;
    *(_DWORD *)(v81 + 184) = v80;
    sub_1D23870();
  }
  else
  {
    *(_QWORD *)(v81 + 176) = 0;
    *(_DWORD *)(v81 + 184) = v80;
  }
}
