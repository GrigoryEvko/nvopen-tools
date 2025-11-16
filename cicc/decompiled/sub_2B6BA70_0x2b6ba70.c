// Function: sub_2B6BA70
// Address: 0x2b6ba70
//
__int64 __fastcall sub_2B6BA70(__int64 a1, unsigned int *a2, __int64 *a3, __int64 a4, __int64 a5, __int64 a6)
{
  __int64 *v8; // rax
  __int64 v9; // rsi
  __int64 v10; // rcx
  __int64 v11; // rdx
  __int64 v12; // rax
  _BYTE **v13; // rbx
  __int64 v14; // r13
  unsigned int *i; // rdx
  _BYTE *v16; // rsi
  __int64 v17; // rbx
  int v18; // r12d
  __int64 v19; // rax
  __int64 v21; // rax
  __int64 v22; // r14
  __int64 *v23; // rbx
  __int64 v24; // rax
  __int64 *v25; // rcx
  __int64 v26; // rdx
  __int64 v27; // rax
  _BYTE *v28; // r13
  __int64 v29; // rax
  char v30; // di
  _DWORD *v31; // rsi
  int v32; // eax
  unsigned int v33; // edx
  _DWORD *v34; // r8
  __int64 v35; // rax
  unsigned int **v36; // r12
  unsigned int *v37; // rax
  __int64 v38; // rsi
  __int64 v39; // rdx
  unsigned int v40; // ebx
  unsigned int *v41; // rdx
  unsigned int v42; // eax
  __int64 v43; // rdx
  __int64 v44; // rdi
  char v45; // al
  __int64 v46; // r13
  __int64 v47; // rax
  char v48; // dl
  int v49; // edi
  unsigned int v50; // eax
  __int64 v51; // rsi
  __int64 v52; // r10
  __int64 v53; // rax
  _BYTE *v54; // rsi
  __int64 v55; // rax
  _BYTE *v56; // rsi
  __int64 v57; // rax
  __int64 v58; // rax
  __int64 v59; // rdx
  __int64 v60; // rdi
  unsigned __int64 v61; // rax
  __int64 v62; // rdi
  char v63; // al
  unsigned int v64; // r8d
  unsigned int v65; // edx
  unsigned __int64 v66; // rax
  __int64 v67; // rdi
  char v68; // al
  unsigned int v69; // r11d
  unsigned int v70; // edx
  __int64 v71; // r8
  __int32 **v72; // rbx
  __int64 *v73; // r12
  __int64 v74; // rsi
  unsigned int v75; // eax
  __int64 v76; // rdx
  __int64 v77; // rcx
  __int64 v78; // rdi
  unsigned __int8 v79; // al
  __int64 v80; // rsi
  unsigned __int64 v81; // rax
  unsigned __int64 v82; // rax
  char v83; // al
  int v84; // r8d
  int v85; // esi
  unsigned __int32 v86; // eax
  __int64 v87; // rax
  __int64 v88; // rax
  int v89; // r11d
  int v90; // r10d
  __int64 *v91; // [rsp+10h] [rbp-D0h]
  __int64 v92; // [rsp+18h] [rbp-C8h]
  unsigned __int8 (__fastcall *v95)(__int64, __int64, __int64, __int64 *, __int64); // [rsp+30h] [rbp-B0h]
  unsigned __int8 v96; // [rsp+38h] [rbp-A8h]
  unsigned int v97; // [rsp+38h] [rbp-A8h]
  unsigned int v98; // [rsp+38h] [rbp-A8h]
  __int64 v99; // [rsp+38h] [rbp-A8h]
  __int64 *v100; // [rsp+40h] [rbp-A0h]
  unsigned __int8 v101; // [rsp+40h] [rbp-A0h]
  __int64 v102; // [rsp+40h] [rbp-A0h]
  __int64 *v103; // [rsp+48h] [rbp-98h]
  __int32 v104; // [rsp+48h] [rbp-98h]
  unsigned __int64 v105; // [rsp+50h] [rbp-90h] BYREF
  unsigned int v106; // [rsp+58h] [rbp-88h]
  __m128i v107; // [rsp+60h] [rbp-80h] BYREF
  __int64 v108; // [rsp+70h] [rbp-70h]
  unsigned int *v109; // [rsp+78h] [rbp-68h]
  __int64 v110; // [rsp+80h] [rbp-60h]
  __int64 v111; // [rsp+88h] [rbp-58h]
  __int64 v112; // [rsp+90h] [rbp-50h]
  __int64 v113; // [rsp+98h] [rbp-48h]
  __int16 v114; // [rsp+A0h] [rbp-40h]

  v95 = (unsigned __int8 (__fastcall *)(__int64, __int64, __int64, __int64 *, __int64))a5;
  v92 = a6;
  if ( !a4 )
  {
    if ( !**(_BYTE **)a1 )
      **(_DWORD **)(a1 + 8) = 1;
    v8 = *(__int64 **)(a1 + 24);
    v9 = *v8;
    v10 = v8[1];
    v11 = v8[2];
    v12 = *(_QWORD *)(a1 + 16);
    v109 = a2;
    v107.m128i_i64[0] = v9;
    v107.m128i_i64[1] = v10;
    v108 = v11;
    v13 = *(_BYTE ***)v12;
    v14 = *(_QWORD *)v12 + 8LL * *(unsigned int *)(v12 + 8);
    if ( *(_QWORD *)v12 != v14 )
    {
      for ( i = a2; ; i = v109 )
      {
        v16 = *v13++;
        sub_2B1EE10(v107.m128i_i64, v16, i);
        if ( (_BYTE **)v14 == v13 )
          break;
      }
    }
LABEL_8:
    ++**(_DWORD **)(a1 + 8);
    v17 = *(_QWORD *)(a1 + 48);
    v18 = *(_DWORD *)(*(_QWORD *)(a1 + 16) + 200LL);
    v19 = *(unsigned int *)(v17 + 8);
    if ( v19 + 1 > (unsigned __int64)*(unsigned int *)(v17 + 12) )
    {
      sub_C8D5F0(*(_QWORD *)(a1 + 48), (const void *)(v17 + 16), v19 + 1, 4u, a5, a6);
      v19 = *(unsigned int *)(v17 + 8);
    }
    *(_DWORD *)(*(_QWORD *)v17 + 4 * v19) = v18;
    ++*(_DWORD *)(v17 + 8);
    LODWORD(a5) = **(unsigned __int8 **)(a1 + 56);
    return (unsigned int)a5;
  }
  v21 = *(_QWORD *)(a1 + 16);
  v22 = *(_QWORD *)(a1 + 24);
  v23 = *(__int64 **)v21;
  v24 = 8LL * *(unsigned int *)(v21 + 8);
  v25 = &v23[(unsigned __int64)v24 / 8];
  v26 = v24 >> 3;
  v27 = v24 >> 5;
  v91 = v25;
  if ( v27 )
  {
    v103 = &v23[4 * v27];
    while ( 1 )
    {
      v28 = (_BYTE *)*v23;
      v29 = *(_QWORD *)(*v23 + 16);
      if ( v29 && !*(_QWORD *)(v29 + 8) || *v28 == 13 )
        goto LABEL_36;
      v25 = *(__int64 **)v22;
      v30 = *(_BYTE *)(*(_QWORD *)v22 + 88LL) & 1;
      if ( v30 )
      {
        v31 = v25 + 12;
        v32 = 3;
      }
      else
      {
        v59 = *((unsigned int *)v25 + 26);
        v31 = (_DWORD *)v25[12];
        if ( !(_DWORD)v59 )
          goto LABEL_78;
        v32 = v59 - 1;
      }
      v33 = v32 & (((unsigned int)v28 >> 9) ^ ((unsigned int)v28 >> 4));
      v34 = &v31[18 * v33];
      a6 = *(_QWORD *)v34;
      if ( *(_BYTE **)v34 == v28 )
        goto LABEL_19;
      v84 = 1;
      while ( a6 != -4096 )
      {
        v90 = v84 + 1;
        v33 = v32 & (v84 + v33);
        v34 = &v31[18 * v33];
        a6 = *(_QWORD *)v34;
        if ( v28 == *(_BYTE **)v34 )
          goto LABEL_19;
        v84 = v90;
      }
      if ( v30 )
      {
        v71 = 72;
        goto LABEL_79;
      }
      v59 = *((unsigned int *)v25 + 26);
LABEL_78:
      v71 = 18 * v59;
LABEL_79:
      v34 = &v31[v71];
LABEL_19:
      v35 = 72;
      if ( !v30 )
        v35 = 18LL * *((unsigned int *)v25 + 26);
      if ( v34 != &v31[v35] && v34[4] > 1u )
        goto LABEL_23;
      v61 = v25[418];
      v62 = *v23;
      v114 = 257;
      v107 = (__m128i)v61;
      v108 = 0;
      v109 = 0;
      v110 = 0;
      v111 = 0;
      v112 = 0;
      v113 = 0;
      v63 = sub_9AC470(v62, &v107, 0);
      if ( **(_BYTE **)(v22 + 8) && v63 )
        goto LABEL_70;
      v64 = *a2;
      v65 = **(_DWORD **)(v22 + 16);
      if ( v65 <= *a2 )
        goto LABEL_70;
      v106 = **(_DWORD **)(v22 + 16);
      if ( v65 <= 0x40 )
      {
        v105 = 0;
LABEL_94:
        if ( v64 > 0x3F || v65 > 0x40 )
          sub_C43C90(&v105, v64, v65);
        else
          v105 |= 0xFFFFFFFFFFFFFFFFLL >> ((unsigned __int8)v64 + 64 - (unsigned __int8)v65) << v64;
        goto LABEL_97;
      }
      v97 = v64;
      sub_C43690((__int64)&v105, 0, 0);
      v65 = v106;
      v64 = v97;
      if ( v97 != v106 )
        goto LABEL_94;
LABEL_97:
      v81 = *(_QWORD *)(*(_QWORD *)v22 + 3344LL);
      v114 = 257;
      v107 = (__m128i)v81;
      v108 = 0;
      v109 = 0;
      v110 = 0;
      v111 = 0;
      v112 = 0;
      v113 = 0;
      if ( (unsigned __int8)sub_9AC230((__int64)v28, (__int64)&v105, &v107, 0) )
      {
        if ( v106 > 0x40 && v105 )
          j_j___libc_free_0_0(v105);
        goto LABEL_36;
      }
      if ( v106 > 0x40 && v105 )
        j_j___libc_free_0_0(v105);
LABEL_70:
      if ( !sub_2B17B70((_DWORD **)v22, v28, a2) )
        goto LABEL_23;
LABEL_36:
      v46 = v23[1];
      a5 = (__int64)(v23 + 1);
      v47 = *(_QWORD *)(v46 + 16);
      if ( v47 && !*(_QWORD *)(v47 + 8) || *(_BYTE *)v46 == 13 )
        goto LABEL_47;
      v25 = *(__int64 **)v22;
      v48 = *(_BYTE *)(*(_QWORD *)v22 + 88LL) & 1;
      if ( v48 )
      {
        a6 = (__int64)(v25 + 12);
        v49 = 3;
      }
      else
      {
        v60 = *((unsigned int *)v25 + 26);
        a6 = v25[12];
        if ( !(_DWORD)v60 )
          goto LABEL_90;
        v49 = v60 - 1;
      }
      v50 = v49 & (((unsigned int)v46 >> 9) ^ ((unsigned int)v46 >> 4));
      v51 = a6 + 72LL * v50;
      v52 = *(_QWORD *)v51;
      if ( v46 == *(_QWORD *)v51 )
        goto LABEL_41;
      v85 = 1;
      while ( v52 != -4096 )
      {
        v89 = v85 + 1;
        v50 = v49 & (v85 + v50);
        v51 = a6 + 72LL * v50;
        v52 = *(_QWORD *)v51;
        if ( v46 == *(_QWORD *)v51 )
          goto LABEL_41;
        v85 = v89;
      }
      if ( v48 )
      {
        v80 = 288;
        goto LABEL_91;
      }
      v60 = *((unsigned int *)v25 + 26);
LABEL_90:
      v80 = 72 * v60;
LABEL_91:
      v51 = a6 + v80;
LABEL_41:
      v53 = 288;
      if ( !v48 )
        v53 = 72LL * *((unsigned int *)v25 + 26);
      if ( v51 != a6 + v53 && *(_DWORD *)(v51 + 16) > 1u )
      {
        ++v23;
        goto LABEL_23;
      }
      v66 = v25[418];
      v67 = v23[1];
      v114 = 257;
      v107 = (__m128i)v66;
      v108 = 0;
      v109 = 0;
      v110 = 0;
      v111 = 0;
      v112 = 0;
      v113 = 0;
      v68 = sub_9AC470(v67, &v107, 0);
      a5 = (__int64)(v23 + 1);
      if ( !**(_BYTE **)(v22 + 8) || !v68 )
      {
        v69 = *a2;
        v70 = **(_DWORD **)(v22 + 16);
        if ( v70 > *a2 )
        {
          v106 = **(_DWORD **)(v22 + 16);
          if ( v70 <= 0x40 )
          {
            v105 = 0;
            goto LABEL_103;
          }
          v98 = v69;
          sub_C43690((__int64)&v105, 0, 0);
          v70 = v106;
          v69 = v98;
          a5 = (__int64)(v23 + 1);
          if ( v98 != v106 )
          {
LABEL_103:
            if ( v69 > 0x3F || v70 > 0x40 )
            {
              v99 = a5;
              sub_C43C90(&v105, v69, v70);
              a5 = v99;
            }
            else
            {
              v105 |= 0xFFFFFFFFFFFFFFFFLL >> ((unsigned __int8)v69 + 64 - (unsigned __int8)v70) << v69;
            }
          }
          v102 = a5;
          v82 = *(_QWORD *)(*(_QWORD *)v22 + 3344LL);
          v108 = 0;
          v107 = (__m128i)v82;
          v109 = 0;
          v110 = 0;
          v111 = 0;
          v112 = 0;
          v113 = 0;
          v114 = 257;
          v83 = sub_9AC230(v46, (__int64)&v105, &v107, 0);
          a5 = v102;
          if ( v83 )
          {
            if ( v106 > 0x40 && v105 )
              j_j___libc_free_0_0(v105);
            goto LABEL_47;
          }
          if ( v106 > 0x40 && v105 )
          {
            j_j___libc_free_0_0(v105);
            a5 = v102;
          }
        }
      }
      v100 = (__int64 *)a5;
      if ( !sub_2B17B70((_DWORD **)v22, (_BYTE *)v46, a2) )
      {
        v23 = v100;
        goto LABEL_23;
      }
LABEL_47:
      v54 = (_BYTE *)v23[2];
      v55 = *((_QWORD *)v54 + 2);
      if ( (!v55 || *(_QWORD *)(v55 + 8)) && !sub_2B1EE10((__int64 *)v22, v54, a2) )
      {
        v23 += 2;
        goto LABEL_23;
      }
      v56 = (_BYTE *)v23[3];
      v57 = *((_QWORD *)v56 + 2);
      if ( (!v57 || *(_QWORD *)(v57 + 8)) && !sub_2B1EE10((__int64 *)v22, v56, a2) )
      {
        v23 += 3;
        goto LABEL_23;
      }
      v23 += 4;
      if ( v23 == v103 )
      {
        v26 = v91 - v23;
        break;
      }
    }
  }
  if ( v26 == 2 )
  {
LABEL_137:
    v88 = *(_QWORD *)(*v23 + 16);
    if ( v88 && !*(_QWORD *)(v88 + 8) || sub_2B1EE10((__int64 *)v22, (_BYTE *)*v23, a2) )
    {
      ++v23;
      goto LABEL_60;
    }
    goto LABEL_23;
  }
  if ( v26 != 3 )
  {
    if ( v26 != 1 )
      goto LABEL_24;
LABEL_60:
    v58 = *(_QWORD *)(*v23 + 16);
    if ( v58 && !*(_QWORD *)(v58 + 8) || sub_2B1EE10((__int64 *)v22, (_BYTE *)*v23, a2) )
      goto LABEL_24;
    goto LABEL_23;
  }
  v87 = *(_QWORD *)(*v23 + 16);
  if ( v87 && !*(_QWORD *)(v87 + 8) || sub_2B1EE10((__int64 *)v22, (_BYTE *)*v23, a2) )
  {
    ++v23;
    goto LABEL_137;
  }
LABEL_23:
  a5 = 0;
  if ( v91 != v23 )
    return (unsigned int)a5;
LABEL_24:
  if ( v95 )
  {
    v36 = *(unsigned int ***)(a1 + 32);
    v37 = *v36;
    v38 = **v36;
    v39 = *v36[1];
    if ( (unsigned int)v38 >= (unsigned int)v39 )
    {
LABEL_133:
      *v37 = v39;
      LODWORD(a5) = 0;
      return (unsigned int)a5;
    }
    v40 = 0;
    while ( !v95(v92, v38, v39, v25, a5) )
    {
      if ( v40 || (v44 = (__int64)v36[2], !**(_BYTE **)(v44 + 8)) )
      {
        v41 = *v36;
        v42 = **v36;
      }
      else
      {
        v45 = sub_2B6ACB0(v44, v38, v43, (__int64)v25, a5, a6);
        v41 = *v36;
        if ( v45 )
          v40 = *v41;
        v42 = *v41;
      }
      *v41 = 2 * v42;
      v37 = *v36;
      v38 = **v36;
      v39 = *v36[1];
      if ( (unsigned int)v38 >= (unsigned int)v39 )
      {
        if ( !v40 )
          goto LABEL_133;
        LODWORD(a5) = 1;
        *v36[3] = 1;
        **v36 = v40;
        return (unsigned int)a5;
      }
    }
  }
  v72 = *(__int32 ***)(a1 + 40);
  v101 = 0;
  v104 = **v72;
  if ( a3 == &a3[a4] )
    goto LABEL_8;
  v73 = a3;
  do
  {
    v74 = *v73;
    v107.m128i_i32[0] = v104;
    v75 = sub_2B69990(
            v72[8],
            v74,
            *(_BYTE *)v72[1],
            (unsigned int *)v72[2],
            v72[3],
            (__int64)v72[4],
            (__int64)v72[5],
            &v107,
            v72[6],
            *(_BYTE *)v72[7]);
    LODWORD(a5) = v75;
    if ( (_BYTE)v75 )
    {
      v86 = **v72;
      if ( v107.m128i_i32[0] >= v86 )
        v86 = v107.m128i_i32[0];
      **v72 = v86;
    }
    else
    {
      if ( !*(_BYTE *)v72[6] )
        return (unsigned int)a5;
      v78 = (__int64)v72[9];
      if ( !**(_BYTE **)(v78 + 8) )
        return (unsigned int)a5;
      v96 = v75;
      v79 = sub_2B6ACB0(v78, v74, v76, v77, v75, a6);
      LODWORD(a5) = v96;
      v101 = v79;
      if ( !v79 )
        return (unsigned int)a5;
    }
    ++v73;
  }
  while ( &a3[a4] != v73 );
  a5 = v101;
  if ( !v101 )
    goto LABEL_8;
  return (unsigned int)a5;
}
