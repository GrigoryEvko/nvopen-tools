// Function: sub_1C650B0
// Address: 0x1c650b0
//
__int64 __fastcall sub_1C650B0(
        __int64 a1,
        __int64 ******a2,
        __int64 a3,
        _QWORD **a4,
        __int64 **a5,
        __int64 a6,
        __m128i a7,
        __m128i a8,
        __int64 a9,
        __int64 a10)
{
  __int64 v11; // r14
  __int64 ****v12; // rbx
  __int64 v13; // r12
  __int64 v14; // rsi
  __int64 v15; // rax
  unsigned __int64 v16; // rcx
  __int64 v17; // r12
  unsigned int *v18; // rax
  __int64 **v19; // rbx
  __int64 **v20; // r13
  __int64 *v21; // r14
  __int64 v22; // rsi
  __int64 v23; // rax
  __int64 v24; // r15
  __int64 **v25; // rdx
  __int64 ****v26; // rbx
  __int64 ***v27; // rax
  __int64 **v28; // r12
  __int64 v29; // r8
  unsigned int v30; // edi
  __int64 *v31; // rcx
  __int64 v32; // rdx
  __int64 *v33; // rbx
  __int64 v34; // r14
  unsigned int v35; // esi
  __int64 v36; // r13
  int v37; // eax
  int v38; // eax
  __int64 v39; // rsi
  unsigned int v40; // r8d
  __int64 *v41; // r10
  __int64 v42; // rcx
  int v43; // edx
  __int64 v44; // rdi
  __int64 v45; // rax
  __int64 *v46; // r13
  __int64 v47; // rsi
  __int64 v48; // rax
  __int64 v49; // rdi
  unsigned __int64 v50; // rcx
  __int64 v51; // rax
  unsigned int v52; // esi
  __int64 *v53; // r12
  __int64 ***v54; // rcx
  __int64 v55; // r8
  unsigned int v56; // edx
  _QWORD *v57; // rax
  __int64 v58; // rdi
  int v59; // eax
  int v60; // edx
  __int64 v61; // rdi
  unsigned int v62; // eax
  __int64 *******v63; // rcx
  __int64 ******v64; // rsi
  int v66; // r11d
  int v67; // ecx
  int v68; // ecx
  __int64 v69; // rsi
  int v70; // ecx
  __int64 *v71; // r9
  int v72; // r11d
  int v73; // r8d
  __int64 v74; // rax
  __int64 *v75; // r8
  __int64 v76; // rax
  int v77; // ecx
  int v78; // r13d
  _QWORD *v79; // r11
  __int64 ***v80; // rdx
  __int64 v81; // rax
  __int64 *v82; // r8
  int v83; // r11d
  int v84; // edi
  unsigned int v85; // r11d
  int v86; // ecx
  int v87; // r8d
  int v88; // edi
  unsigned int v89; // r11d
  __int64 v90; // [rsp+8h] [rbp-110h]
  unsigned int v91; // [rsp+10h] [rbp-108h]
  __int64 *v93; // [rsp+30h] [rbp-E8h]
  __int64 *v94; // [rsp+40h] [rbp-D8h]
  __int64 v96; // [rsp+50h] [rbp-C8h]
  __int64 v97; // [rsp+58h] [rbp-C0h]
  int v98; // [rsp+64h] [rbp-B4h]
  __int64 v99; // [rsp+68h] [rbp-B0h]
  __int64 v101; // [rsp+78h] [rbp-A0h]
  __int64 *v104; // [rsp+90h] [rbp-88h]
  __int64 v105; // [rsp+98h] [rbp-80h]
  __int64 v107; // [rsp+A0h] [rbp-78h]
  unsigned __int64 v108; // [rsp+B0h] [rbp-68h] BYREF
  __int64 v109; // [rsp+B8h] [rbp-60h] BYREF
  __int64 v110; // [rsp+C0h] [rbp-58h] BYREF
  __int64 *v111; // [rsp+C8h] [rbp-50h] BYREF
  __int64 v112; // [rsp+D0h] [rbp-48h]
  __int64 v113; // [rsp+D8h] [rbp-40h] BYREF
  __int64 v114; // [rsp+E0h] [rbp-38h]

  v11 = a10;
  v12 = **a2;
  sub_1C620D0(
    (_QWORD *)a1,
    (unsigned int *)*v12,
    (__int64 **)v12[1],
    &v108,
    (unsigned __int64 *)&v109,
    *(_QWORD *)(a1 + 200),
    0);
  v93 = a5[1];
  v94 = *a5;
  v13 = (***v12)[1];
  if ( *(_WORD *)(v13 + 24) != 4 )
    v13 = 0;
  if ( *a5 == a5[1] )
  {
    v110 = sub_38767A0(a6, **(_QWORD **)(v13 + 32), 0, **a4);
    v14 = v110;
    v82 = a5[1];
    if ( v82 == a5[2] )
    {
      sub_1287830((__int64)a5, a5[1], &v110);
      v14 = v110;
    }
    else
    {
      if ( v82 )
      {
        *v82 = v110;
        v82 = a5[1];
      }
      a5[1] = v82 + 1;
    }
  }
  else
  {
    v14 = *v94;
    v110 = *v94;
  }
  v99 = sub_145DC80(*(_QWORD *)(a1 + 184), v14);
  v96 = *(_QWORD *)(*(_QWORD *)(v13 + 32) + 8LL);
  v15 = sub_13A5B00(*(_QWORD *)(a1 + 184), v99, v96, 0, 0);
  v16 = v108;
  v104 = (__int64 *)v15;
  if ( *(_BYTE *)(v108 + 16) == 18 )
  {
    if ( v108 == *(_QWORD *)((***v12)[2] + 40) )
      v16 = (***v12)[2];
    else
      v16 = sub_157EBA0(v108);
  }
  v17 = sub_38767A0(a6, v104, 0, v16);
  sub_1C538E0(a9, (__int64 *)**a2)[1] = v17;
  v18 = (unsigned int *)*v12;
  v19 = **v12;
  v20 = &v19[v18[2]];
  if ( v19 != v20 )
  {
    do
    {
      v21 = *v19;
      v22 = v17;
      if ( !sub_14560B0(**v19) )
      {
        v23 = sub_13A5B00(*(_QWORD *)(a1 + 184), (__int64)v104, *v21, 0, 0);
        v22 = sub_38767A0(a6, v23, 0, v21[2]);
      }
      ++v19;
      sub_1C51F30((__int64 ***)v21[3], v22, v21[2]);
    }
    while ( v19 != v20 );
    v11 = a10;
  }
  if ( v94 == v93 )
  {
    v81 = sub_38767A0(a6, a3, 0, (*a4)[1]);
    v90 = sub_145DC80(*(_QWORD *)(a1 + 184), v81);
  }
  else
  {
    v90 = 0;
  }
  v24 = v11;
  v98 = 0;
  v101 = 0;
  v25 = (__int64 **)*a2;
  if ( a2[1] != *a2 )
  {
    while ( 1 )
    {
      v97 = v101;
      sub_1C620D0(
        (_QWORD *)a1,
        (unsigned int *)*v25[v101],
        (__int64 **)v25[v101][1],
        &v108,
        (unsigned __int64 *)&v109,
        *(_QWORD *)(a1 + 200),
        0);
      v101 = (unsigned int)++v98;
      if ( v94 == v93 )
      {
        v76 = sub_13A5B00(*(_QWORD *)(a1 + 184), v99, v90, 0, 0);
        v110 = 0;
        v99 = v76;
      }
      else
      {
        v110 = (*a5)[v98];
      }
      v26 = (*a2)[v97];
      v27 = v26[1];
      v28 = *v27;
      v105 = (__int64)&(*v27)[*((unsigned int *)v27 + 2)];
      if ( *v27 != (__int64 **)v105 )
        break;
      v51 = a9;
      v107 = 0;
      v53 = (__int64 *)(v26 + 1);
      v52 = *(_DWORD *)(a9 + 24);
      if ( !v52 )
        goto LABEL_70;
LABEL_43:
      v54 = v26[1];
      v55 = *(_QWORD *)(v51 + 8);
      v56 = (v52 - 1) & (((unsigned int)v54 >> 9) ^ ((unsigned int)v54 >> 4));
      v57 = (_QWORD *)(v55 + 16LL * v56);
      v58 = *v57;
      if ( v54 != (__int64 ***)*v57 )
      {
        v78 = 1;
        v79 = 0;
        while ( v58 != -8 )
        {
          if ( v58 == -16 && !v79 )
            v79 = v57;
          v56 = (v52 - 1) & (v78 + v56);
          v57 = (_QWORD *)(v55 + 16LL * v56);
          v58 = *v57;
          if ( v54 == (__int64 ***)*v57 )
            goto LABEL_44;
          ++v78;
        }
        if ( v79 )
          v57 = v79;
        ++*(_QWORD *)a9;
        v77 = *(_DWORD *)(a9 + 16) + 1;
        if ( 4 * v77 >= 3 * v52 )
          goto LABEL_71;
        if ( v52 - *(_DWORD *)(a9 + 20) - v77 <= v52 >> 3 )
        {
LABEL_72:
          sub_1C53730(a9, v52);
          sub_1C502F0(a9, v53, &v111);
          v57 = v111;
          v77 = *(_DWORD *)(a9 + 16) + 1;
        }
        *(_DWORD *)(a9 + 16) = v77;
        if ( *v57 != -8 )
          --*(_DWORD *)(a9 + 20);
        v80 = v26[1];
        v57[1] = 0;
        *v57 = v80;
      }
LABEL_44:
      v57[1] = v107;
      v25 = (__int64 **)*a2;
      if ( a2[1] - *a2 == v98 )
        goto LABEL_45;
    }
    v107 = 0;
    while ( 1 )
    {
      v33 = *v28;
      if ( !v107 )
      {
        v47 = v110;
        if ( !v110 )
        {
          v110 = sub_38767A0(a6, v99, 0, (*a4)[v98]);
          v47 = v110;
          v75 = a5[1];
          if ( v75 == a5[2] )
          {
            sub_1287830((__int64)a5, a5[1], &v110);
            v47 = v110;
          }
          else
          {
            if ( v75 )
            {
              *v75 = v110;
              v75 = a5[1];
            }
            a5[1] = v75 + 1;
          }
        }
        v48 = sub_145DC80(*(_QWORD *)(a1 + 184), v47);
        v49 = *(_QWORD *)(a1 + 184);
        v99 = v48;
        v113 = v48;
        v111 = &v113;
        v114 = v96;
        v112 = 0x200000002LL;
        v104 = sub_147DD40(v49, (__int64 *)&v111, 0, 0, a7, a8);
        if ( v111 != &v113 )
          _libc_free((unsigned __int64)v111);
        v50 = v109;
        if ( *(_BYTE *)(v109 + 16) == 18 )
        {
          if ( v109 == *(_QWORD *)(v33[2] + 40) )
            v50 = v33[2];
          else
            v50 = sub_157EBA0(v109);
        }
        v107 = sub_38767A0(a6, v104, 0, v50);
      }
      v34 = v107;
      if ( !sub_14560B0(*v33) )
      {
        v44 = *(_QWORD *)(a1 + 184);
        v45 = *v33;
        v111 = &v113;
        v113 = (__int64)v104;
        v114 = v45;
        v112 = 0x200000002LL;
        v46 = sub_147DD40(v44, (__int64 *)&v111, 0, 0, a7, a8);
        if ( v111 != &v113 )
          _libc_free((unsigned __int64)v111);
        v34 = sub_38767A0(a6, v46, 0, v33[2]);
      }
      v35 = *(_DWORD *)(v24 + 24);
      v36 = v33[2];
      if ( !v35 )
        break;
      v29 = *(_QWORD *)(v24 + 8);
      v30 = (v35 - 1) & (((unsigned int)v36 >> 4) ^ ((unsigned int)v36 >> 9));
      v31 = (__int64 *)(v29 + 8LL * v30);
      v32 = *v31;
      if ( v36 != *v31 )
      {
        v66 = 1;
        v41 = 0;
        while ( v32 != -8 )
        {
          if ( v41 || v32 != -16 )
            v31 = v41;
          v30 = (v35 - 1) & (v66 + v30);
          v32 = *(_QWORD *)(v29 + 8LL * v30);
          if ( v36 == v32 )
            goto LABEL_22;
          ++v66;
          v41 = v31;
          v31 = (__int64 *)(v29 + 8LL * v30);
        }
        if ( !v41 )
          v41 = v31;
        v67 = *(_DWORD *)(v24 + 16);
        ++*(_QWORD *)v24;
        v43 = v67 + 1;
        if ( 4 * (v67 + 1) < 3 * v35 )
        {
          if ( v35 - *(_DWORD *)(v24 + 20) - v43 <= v35 >> 3 )
          {
            v91 = ((unsigned int)v36 >> 4) ^ ((unsigned int)v36 >> 9);
            sub_1467110(v24, v35);
            v68 = *(_DWORD *)(v24 + 24);
            if ( !v68 )
            {
LABEL_116:
              ++*(_DWORD *)(v24 + 16);
              BUG();
            }
            v69 = *(_QWORD *)(v24 + 8);
            v70 = v68 - 1;
            v71 = 0;
            v72 = 1;
            v41 = (__int64 *)(v69 + 8LL * (v70 & v91));
            v73 = v70 & v91;
            v43 = *(_DWORD *)(v24 + 16) + 1;
            v74 = *v41;
            if ( v36 != *v41 )
            {
              while ( v74 != -8 )
              {
                if ( !v71 && v74 == -16 )
                  v71 = v41;
                v88 = v72 + 1;
                v89 = v70 & (v73 + v72);
                v41 = (__int64 *)(v69 + 8LL * v89);
                v73 = v89;
                v74 = *v41;
                if ( v36 == *v41 )
                  goto LABEL_29;
                v72 = v88;
              }
LABEL_60:
              if ( v71 )
                v41 = v71;
            }
          }
LABEL_29:
          *(_DWORD *)(v24 + 16) = v43;
          if ( *v41 != -8 )
            --*(_DWORD *)(v24 + 20);
          *v41 = v36;
          v32 = v33[2];
          goto LABEL_22;
        }
LABEL_27:
        sub_1467110(v24, 2 * v35);
        v37 = *(_DWORD *)(v24 + 24);
        if ( !v37 )
          goto LABEL_116;
        v38 = v37 - 1;
        v39 = *(_QWORD *)(v24 + 8);
        v40 = v38 & (((unsigned int)v36 >> 9) ^ ((unsigned int)v36 >> 4));
        v41 = (__int64 *)(v39 + 8LL * v40);
        v42 = *v41;
        v43 = *(_DWORD *)(v24 + 16) + 1;
        if ( v36 != *v41 )
        {
          v83 = 1;
          v71 = 0;
          while ( v42 != -8 )
          {
            if ( v42 == -16 && !v71 )
              v71 = v41;
            v84 = v83 + 1;
            v85 = v38 & (v40 + v83);
            v41 = (__int64 *)(v39 + 8LL * v85);
            v40 = v85;
            v42 = *v41;
            if ( v36 == *v41 )
              goto LABEL_29;
            v83 = v84;
          }
          goto LABEL_60;
        }
        goto LABEL_29;
      }
LABEL_22:
      ++v28;
      sub_1C51F30((__int64 ***)v33[3], v34, v32);
      if ( (__int64 **)v105 == v28 )
      {
        v26 = (*a2)[v97];
        v51 = a9;
        v52 = *(_DWORD *)(a9 + 24);
        v53 = (__int64 *)(v26 + 1);
        if ( v52 )
          goto LABEL_43;
LABEL_70:
        ++*(_QWORD *)a9;
LABEL_71:
        v52 *= 2;
        goto LABEL_72;
      }
    }
    ++*(_QWORD *)v24;
    goto LABEL_27;
  }
LABEL_45:
  v59 = *(_DWORD *)(a1 + 256);
  if ( v59 )
  {
    v60 = v59 - 1;
    v61 = *(_QWORD *)(a1 + 240);
    v62 = (v59 - 1) & (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4));
    v63 = (__int64 *******)(v61 + 8LL * v62);
    v64 = *v63;
    if ( *v63 == a2 )
    {
LABEL_47:
      *v63 = (__int64 ******)-16LL;
      --*(_DWORD *)(a1 + 248);
      ++*(_DWORD *)(a1 + 252);
    }
    else
    {
      v86 = 1;
      while ( v64 != (__int64 ******)-8LL )
      {
        v87 = v86 + 1;
        v62 = v60 & (v86 + v62);
        v63 = (__int64 *******)(v61 + 8LL * v62);
        v64 = *v63;
        if ( *v63 == a2 )
          goto LABEL_47;
        v86 = v87;
      }
    }
  }
  if ( *a2 )
    j_j___libc_free_0(*a2, (char *)a2[2] - (char *)*a2);
  return j_j___libc_free_0(a2, 24);
}
