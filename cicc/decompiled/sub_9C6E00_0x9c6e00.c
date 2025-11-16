// Function: sub_9C6E00
// Address: 0x9c6e00
//
_QWORD *__fastcall sub_9C6E00(
        __int64 a1,
        int a2,
        int a3,
        int a4,
        __int64 a5,
        __int64 a6,
        _QWORD *a7,
        __int64 *a8,
        __int64 *a9,
        __int64 *a10,
        __int64 *a11,
        __int64 *a12,
        __int64 *a13,
        __int64 a14)
{
  __int64 v14; // rbx
  _QWORD *v15; // r10
  __int64 v16; // rsi
  __int64 v17; // rdi
  unsigned int v18; // r8d
  __int64 v19; // rdx
  __int64 v20; // r8
  __int64 v21; // rax
  __int64 v22; // r9
  __int64 v23; // rax
  __int64 v24; // r10
  __int64 v25; // rcx
  __int64 v26; // rax
  __int64 v27; // r12
  __int64 v28; // rax
  __int64 v29; // r13
  __int64 v30; // r14
  _QWORD *v31; // rax
  _QWORD *v32; // r14
  __int64 v33; // r15
  __int64 v34; // rbx
  __int64 v35; // r15
  __int64 v36; // rdi
  __int64 v37; // r15
  __int64 v38; // rbx
  __int64 v39; // r15
  __int64 v40; // rdi
  __int64 v41; // rdi
  __int64 v42; // rdi
  __int64 i; // r14
  __int64 v44; // rdi
  __int64 j; // r13
  __int64 v46; // rdi
  _QWORD *v47; // rax
  __int64 v48; // rdx
  __int64 v49; // rdx
  __int64 v50; // rdx
  _QWORD *v51; // r15
  __int64 v52; // r14
  __int64 v53; // r13
  __int64 v54; // r12
  __int64 v55; // r15
  __int64 v56; // rdi
  __int64 v57; // rdi
  __int64 v58; // rdi
  __int64 v59; // rdi
  _QWORD *v60; // rax
  __int64 v61; // rdx
  __int64 v62; // rdx
  __int64 v63; // rdx
  __int64 *v64; // r14
  __int64 v65; // r13
  __int64 v66; // r12
  __int64 v67; // rdi
  __int64 v68; // rdi
  _QWORD *result; // rax
  _QWORD *v70; // rdx
  __int64 v71; // rdx
  __int64 v72; // rdx
  _QWORD *v73; // r14
  _QWORD *v74; // rbx
  _QWORD *v75; // r13
  __int64 *v76; // r15
  __int64 *v77; // r12
  __int64 v78; // rdi
  __int64 v79; // r15
  __int64 v80; // r12
  __int64 v81; // rdi
  __int64 v82; // r9
  int v83; // r8d
  size_t v84; // rdx
  int v85; // edx
  __int64 v86; // rcx
  __int64 v87; // [rsp+8h] [rbp-A8h]
  __int64 v88; // [rsp+10h] [rbp-A0h]
  __int64 v89; // [rsp+10h] [rbp-A0h]
  __int64 v90; // [rsp+10h] [rbp-A0h]
  __int64 v91; // [rsp+18h] [rbp-98h]
  __int64 v92; // [rsp+18h] [rbp-98h]
  __int64 v93; // [rsp+20h] [rbp-90h]
  __int64 v94; // [rsp+20h] [rbp-90h]
  __int64 v95; // [rsp+28h] [rbp-88h]
  __int64 v96; // [rsp+28h] [rbp-88h]
  __int64 v97; // [rsp+30h] [rbp-80h]
  __int64 v98; // [rsp+30h] [rbp-80h]
  __int64 v99; // [rsp+38h] [rbp-78h]
  __int64 v100; // [rsp+38h] [rbp-78h]
  __int64 v101; // [rsp+58h] [rbp-58h]
  __int64 v102; // [rsp+60h] [rbp-50h]
  __int64 v103; // [rsp+68h] [rbp-48h]
  _QWORD *v104; // [rsp+68h] [rbp-48h]
  __int64 v105; // [rsp+68h] [rbp-48h]
  __int64 v106; // [rsp+70h] [rbp-40h]
  __int64 v107; // [rsp+70h] [rbp-40h]
  int v108; // [rsp+70h] [rbp-40h]
  int v109; // [rsp+70h] [rbp-40h]
  __int64 v110; // [rsp+78h] [rbp-38h]
  _QWORD *v111; // [rsp+78h] [rbp-38h]
  int v112; // [rsp+78h] [rbp-38h]
  __int64 v113; // [rsp+78h] [rbp-38h]
  int v114; // [rsp+78h] [rbp-38h]

  v14 = a1;
  *(_DWORD *)(a1 + 8) = 1;
  *(_DWORD *)(a1 + 12) = a2;
  v15 = a7;
  *(_QWORD *)(a1 + 16) = 0;
  *(_QWORD *)(a1 + 24) = 0;
  *(_QWORD *)(a1 + 32) = 0;
  *(_QWORD *)(a1 + 48) = 0;
  *(_QWORD *)a1 = &unk_49D9770;
  *(_QWORD *)(a1 + 40) = a1 + 56;
  if ( *(_DWORD *)(a5 + 8) )
  {
    v105 = a6;
    v109 = a4;
    v114 = a3;
    sub_9C2F00(a1 + 40, (char **)a5);
    v15 = a7;
    a6 = v105;
    a4 = v109;
    a3 = v114;
  }
  v16 = a1 + 80;
  *(_DWORD *)(a1 + 56) = a3;
  v17 = a1 + 64;
  *(_DWORD *)(v14 + 60) = a4;
  *(_QWORD *)(v14 + 64) = v14 + 80;
  *(_QWORD *)v14 = &unk_49D97B0;
  *(_QWORD *)(v14 + 72) = 0;
  v18 = *(_DWORD *)(a6 + 8);
  if ( v18 && v17 != a6 )
  {
    if ( *(_QWORD *)a6 == a6 + 16 )
    {
      v104 = v15;
      v107 = a6;
      v112 = *(_DWORD *)(a6 + 8);
      sub_C8D5F0(v17, v16, v18, 16);
      v82 = v107;
      v83 = v112;
      v15 = v104;
      v84 = 16LL * *(unsigned int *)(v107 + 8);
      if ( v84 )
      {
        v16 = *(_QWORD *)v107;
        v108 = v112;
        v113 = v82;
        memcpy(*(void **)(v14 + 64), (const void *)v16, v84);
        v15 = v104;
        v83 = v108;
        v82 = v113;
      }
      *(_DWORD *)(v14 + 72) = v83;
      *(_DWORD *)(v82 + 8) = 0;
    }
    else
    {
      *(_QWORD *)(v14 + 64) = *(_QWORD *)a6;
      v85 = *(_DWORD *)(a6 + 12);
      *(_DWORD *)(v14 + 72) = v18;
      *(_DWORD *)(v14 + 76) = v85;
      *(_QWORD *)a6 = a6 + 16;
      *(_QWORD *)(a6 + 8) = 0;
    }
  }
  *(_QWORD *)(v14 + 80) = 0;
  *(_QWORD *)(v14 + 88) = 0;
  *(_QWORD *)(v14 + 96) = 0;
  *(_QWORD *)(v14 + 104) = 0;
  v19 = v15[1];
  v103 = *v15;
  if ( *v15 != v19 || *a8 != a8[1] || *a9 != a9[1] || *a10 != a10[1] || *a11 != a11[1] )
  {
    *v15 = 0;
    v20 = v15[2];
    v15[2] = 0;
    v15[1] = 0;
    v21 = *a8;
    v22 = a8[1];
    *a8 = 0;
    a8[1] = 0;
    v16 = a8[2];
    v101 = v21;
    a8[2] = 0;
    v23 = *a9;
    v24 = a9[1];
    v25 = a9[2];
    a9[1] = 0;
    a9[2] = 0;
    *a9 = 0;
    v102 = v23;
    v26 = *a10;
    v87 = v19;
    v27 = a10[1];
    v88 = v20;
    v91 = v22;
    v93 = v16;
    v95 = v24;
    v97 = v25;
    v99 = a10[2];
    a10[2] = 0;
    a10[1] = 0;
    *a10 = 0;
    v106 = v26;
    v28 = *a11;
    v29 = a11[1];
    v30 = a11[2];
    a11[1] = 0;
    a11[2] = 0;
    *a11 = 0;
    v110 = v28;
    v31 = (_QWORD *)sub_22077B0(120);
    if ( v31 )
    {
      v31[1] = v87;
      v31[8] = v97;
      v31[3] = v101;
      v31[9] = v106;
      *v31 = v103;
      v31[2] = v88;
      v31[4] = v91;
      v31[5] = v16;
      v31[6] = v102;
      v31[7] = v95;
      v31[11] = v99;
      v31[12] = v110;
      v31[14] = v30;
      v100 = 0;
      v94 = 0;
      v96 = 0;
      v98 = 0;
      v92 = 0;
      v110 = 0;
      v106 = 0;
      v102 = 0;
      v101 = 0;
      v103 = 0;
      v31[10] = v27;
      v27 = 0;
      v31[13] = v29;
      v29 = 0;
    }
    else
    {
      v16 -= v101;
      v86 = v97 - v102;
      v98 = v93 - v101;
      v92 = v88 - v103;
      v96 = v86;
      v94 = v30 - v110;
      v100 = v99 - v106;
    }
    v32 = *(_QWORD **)(v14 + 80);
    *(_QWORD *)(v14 + 80) = v31;
    if ( v32 )
    {
      v33 = v32[12];
      if ( v32[13] != v33 )
      {
        v89 = v14;
        v34 = v32[12];
        v35 = v32[13];
        do
        {
          v36 = *(_QWORD *)(v34 + 16);
          if ( v36 )
            j_j___libc_free_0(v36, *(_QWORD *)(v34 + 32) - v36);
          v34 += 40;
        }
        while ( v35 != v34 );
        v14 = v89;
        v33 = v32[12];
      }
      if ( v33 )
        j_j___libc_free_0(v33, v32[14] - v33);
      v37 = v32[9];
      if ( v32[10] != v37 )
      {
        v90 = v14;
        v38 = v32[9];
        v39 = v32[10];
        do
        {
          v40 = *(_QWORD *)(v38 + 16);
          if ( v40 )
            j_j___libc_free_0(v40, *(_QWORD *)(v38 + 32) - v40);
          v38 += 40;
        }
        while ( v39 != v38 );
        v14 = v90;
        v37 = v32[9];
      }
      if ( v37 )
        j_j___libc_free_0(v37, v32[11] - v37);
      v41 = v32[6];
      if ( v41 )
        j_j___libc_free_0(v41, v32[8] - v41);
      v42 = v32[3];
      if ( v42 )
        j_j___libc_free_0(v42, v32[5] - v42);
      if ( *v32 )
        j_j___libc_free_0(*v32, v32[2] - *v32);
      v16 = 120;
      j_j___libc_free_0(v32, 120);
    }
    for ( i = v110; v29 != i; i += 40 )
    {
      v44 = *(_QWORD *)(i + 16);
      if ( v44 )
      {
        v16 = *(_QWORD *)(i + 32) - v44;
        j_j___libc_free_0(v44, v16);
      }
    }
    if ( v110 )
    {
      v16 = v94;
      j_j___libc_free_0(v110, v94);
    }
    for ( j = v106; j != v27; j += 40 )
    {
      v46 = *(_QWORD *)(j + 16);
      if ( v46 )
      {
        v16 = *(_QWORD *)(j + 32) - v46;
        j_j___libc_free_0(v46, v16);
      }
    }
    if ( v106 )
    {
      v16 = v100;
      j_j___libc_free_0(v106, v100);
    }
    if ( v102 )
    {
      v16 = v96;
      j_j___libc_free_0(v102, v96);
    }
    if ( v101 )
    {
      v16 = v98;
      j_j___libc_free_0(v101, v98);
    }
    if ( v103 )
    {
      v16 = v92;
      j_j___libc_free_0(v103, v92);
    }
  }
  if ( a12[1] != *a12 )
  {
    v47 = (_QWORD *)sub_22077B0(24);
    if ( v47 )
    {
      v48 = *a12;
      *a12 = 0;
      *v47 = v48;
      v49 = a12[1];
      a12[1] = 0;
      v47[1] = v49;
      v50 = a12[2];
      a12[2] = 0;
      v47[2] = v50;
    }
    v51 = *(_QWORD **)(v14 + 88);
    *(_QWORD *)(v14 + 88) = v47;
    if ( v51 )
    {
      v52 = v51[1];
      v53 = *v51;
      if ( v52 != *v51 )
      {
        v111 = v51;
        do
        {
          v54 = *(_QWORD *)(v53 + 48);
          v55 = *(_QWORD *)(v53 + 40);
          if ( v54 != v55 )
          {
            do
            {
              if ( *(_DWORD *)(v55 + 40) > 0x40u )
              {
                v56 = *(_QWORD *)(v55 + 32);
                if ( v56 )
                  j_j___libc_free_0_0(v56);
              }
              if ( *(_DWORD *)(v55 + 24) > 0x40u )
              {
                v57 = *(_QWORD *)(v55 + 16);
                if ( v57 )
                  j_j___libc_free_0_0(v57);
              }
              v55 += 48;
            }
            while ( v54 != v55 );
            v55 = *(_QWORD *)(v53 + 40);
          }
          if ( v55 )
            j_j___libc_free_0(v55, *(_QWORD *)(v53 + 56) - v55);
          if ( *(_DWORD *)(v53 + 32) > 0x40u )
          {
            v58 = *(_QWORD *)(v53 + 24);
            if ( v58 )
              j_j___libc_free_0_0(v58);
          }
          if ( *(_DWORD *)(v53 + 16) > 0x40u )
          {
            v59 = *(_QWORD *)(v53 + 8);
            if ( v59 )
              j_j___libc_free_0_0(v59);
          }
          v53 += 64;
        }
        while ( v52 != v53 );
        v51 = v111;
        v53 = *v111;
      }
      if ( v53 )
        j_j___libc_free_0(v53, v51[2] - v53);
      v16 = 24;
      j_j___libc_free_0(v51, 24);
    }
  }
  if ( a13[1] != *a13 )
  {
    v60 = (_QWORD *)sub_22077B0(24);
    if ( v60 )
    {
      v61 = *a13;
      *a13 = 0;
      *v60 = v61;
      v62 = a13[1];
      a13[1] = 0;
      v60[1] = v62;
      v63 = a13[2];
      a13[2] = 0;
      v60[2] = v63;
    }
    v64 = *(__int64 **)(v14 + 96);
    *(_QWORD *)(v14 + 96) = v60;
    if ( v64 )
    {
      v65 = v64[1];
      v66 = *v64;
      if ( v65 != *v64 )
      {
        do
        {
          v67 = *(_QWORD *)(v66 + 72);
          if ( v67 != v66 + 88 )
            _libc_free(v67, v16);
          v68 = *(_QWORD *)(v66 + 8);
          if ( v68 != v66 + 24 )
            _libc_free(v68, v16);
          v66 += 136;
        }
        while ( v65 != v66 );
        v66 = *v64;
      }
      if ( v66 )
        j_j___libc_free_0(v66, v64[2] - v66);
      v16 = 24;
      j_j___libc_free_0(v64, 24);
    }
  }
  result = *(_QWORD **)a14;
  if ( *(_QWORD *)(a14 + 8) != *(_QWORD *)a14 )
  {
    result = (_QWORD *)sub_22077B0(24);
    if ( result )
    {
      v70 = *(_QWORD **)a14;
      *(_QWORD *)a14 = 0;
      *result = v70;
      v71 = *(_QWORD *)(a14 + 8);
      *(_QWORD *)(a14 + 8) = 0;
      result[1] = v71;
      v72 = *(_QWORD *)(a14 + 16);
      *(_QWORD *)(a14 + 16) = 0;
      result[2] = v72;
    }
    v73 = *(_QWORD **)(v14 + 104);
    *(_QWORD *)(v14 + 104) = result;
    if ( v73 )
    {
      v74 = (_QWORD *)v73[1];
      v75 = (_QWORD *)*v73;
      if ( v74 != (_QWORD *)*v73 )
      {
        do
        {
          v76 = (__int64 *)v75[12];
          v77 = (__int64 *)v75[11];
          if ( v76 != v77 )
          {
            do
            {
              v78 = *v77;
              if ( *v77 )
              {
                v16 = v77[2] - v78;
                j_j___libc_free_0(v78, v16);
              }
              v77 += 3;
            }
            while ( v76 != v77 );
            v77 = (__int64 *)v75[11];
          }
          if ( v77 )
          {
            v16 = v75[13] - (_QWORD)v77;
            j_j___libc_free_0(v77, v16);
          }
          v79 = v75[9];
          v80 = v75[8];
          if ( v79 != v80 )
          {
            do
            {
              v81 = *(_QWORD *)(v80 + 8);
              if ( v81 != v80 + 24 )
                _libc_free(v81, v16);
              v80 += 72;
            }
            while ( v79 != v80 );
            v80 = v75[8];
          }
          if ( v80 )
          {
            v16 = v75[10] - v80;
            j_j___libc_free_0(v80, v16);
          }
          if ( (_QWORD *)*v75 != v75 + 3 )
            _libc_free(*v75, v16);
          v75 += 14;
        }
        while ( v74 != v75 );
        v75 = (_QWORD *)*v73;
      }
      if ( v75 )
        j_j___libc_free_0(v75, v73[2] - (_QWORD)v75);
      return (_QWORD *)j_j___libc_free_0(v73, 24);
    }
  }
  return result;
}
