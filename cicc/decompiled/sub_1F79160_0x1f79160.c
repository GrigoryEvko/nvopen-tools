// Function: sub_1F79160
// Address: 0x1f79160
//
__int64 *__fastcall sub_1F79160(
        _BYTE **a1,
        __int64 a2,
        __int64 a3,
        __int64 a4,
        unsigned __int16 a5,
        int a6,
        double a7,
        double a8,
        __m128i a9)
{
  _BYTE **v10; // r12
  __int64 v11; // rbx
  __int64 v13; // r15
  __int64 v14; // rax
  __int64 v15; // rdx
  int v16; // ecx
  __int16 v17; // r8
  int v18; // r9d
  void *v19; // rax
  void *v20; // rdx
  void *v21; // rax
  __int64 v22; // rax
  void *v23; // rax
  void *v24; // rdx
  __int64 v25; // r8
  void *v26; // rax
  __int64 v27; // rdi
  __int64 v28; // rdi
  bool v29; // al
  __int64 *v30; // r13
  unsigned __int64 v31; // rax
  __int16 *v32; // rdx
  __int64 v33; // rdi
  bool v34; // al
  __int64 *v35; // r13
  __int64 v36; // rdx
  unsigned __int64 v37; // rax
  __int16 *v38; // rdx
  __int64 v39; // r13
  __int64 v40; // r15
  __int64 v41; // rbx
  __int64 v42; // rdx
  __int64 v43; // r15
  __int64 v44; // rbx
  __int64 v45; // rdi
  __int64 *v46; // r13
  __int64 v47; // rax
  __int64 v48; // rdx
  __int64 v49; // rdx
  __int64 v50; // r12
  __int64 v51; // r15
  __int64 v52; // r13
  __int64 v53; // rbx
  __int64 v54; // rdx
  __int64 v55; // r13
  __int64 v56; // r15
  __int64 v57; // rbx
  __int128 v58; // [rsp-30h] [rbp-100h]
  __int128 v59; // [rsp-30h] [rbp-100h]
  __int128 v60; // [rsp-20h] [rbp-F0h]
  __int128 v61; // [rsp-20h] [rbp-F0h]
  __int128 v62; // [rsp-10h] [rbp-E0h]
  __int128 v63; // [rsp-10h] [rbp-E0h]
  void *v64; // [rsp+0h] [rbp-D0h]
  __int64 v65; // [rsp+0h] [rbp-D0h]
  void *v66; // [rsp+8h] [rbp-C8h]
  void *v67; // [rsp+8h] [rbp-C8h]
  __int16 *v68; // [rsp+10h] [rbp-C0h]
  void *v69; // [rsp+10h] [rbp-C0h]
  __int64 v70; // [rsp+18h] [rbp-B8h]
  __int64 v71; // [rsp+20h] [rbp-B0h]
  __int16 *v72; // [rsp+20h] [rbp-B0h]
  __int64 v73; // [rsp+28h] [rbp-A8h]
  void *v74; // [rsp+30h] [rbp-A0h]
  void *v75; // [rsp+30h] [rbp-A0h]
  __int64 v76; // [rsp+38h] [rbp-98h]
  __int64 v77; // [rsp+38h] [rbp-98h]
  __int64 v78; // [rsp+38h] [rbp-98h]
  __int64 v79; // [rsp+38h] [rbp-98h]
  __int64 v80; // [rsp+38h] [rbp-98h]
  _BYTE **v81; // [rsp+38h] [rbp-98h]
  __int64 v82; // [rsp+38h] [rbp-98h]
  __int64 v83; // [rsp+40h] [rbp-90h]
  __int64 v84; // [rsp+48h] [rbp-88h]
  __int64 v85; // [rsp+48h] [rbp-88h]
  __int64 v86; // [rsp+48h] [rbp-88h]
  bool v87; // [rsp+5Dh] [rbp-73h]
  bool v88; // [rsp+5Dh] [rbp-73h]
  __int64 v90[4]; // [rsp+60h] [rbp-70h] BYREF
  char v91[8]; // [rsp+80h] [rbp-50h] BYREF
  void *v92; // [rsp+88h] [rbp-48h] BYREF
  __int64 v93; // [rsp+90h] [rbp-40h]

  if ( *(_WORD *)(a2 + 24) != 77 )
    return 0;
  v10 = a1;
  v11 = a2;
  v13 = a4;
  if ( !**a1 )
  {
    v14 = *(_QWORD *)(a2 + 48);
    if ( !v14 || *(_QWORD *)(v14 + 32) )
      return 0;
  }
  v76 = sub_1D23470(**(_QWORD **)(a2 + 32), *(_QWORD *)(*(_QWORD *)(a2 + 32) + 8LL), a3, a4, a5, a6);
  if ( !v76 )
    goto LABEL_15;
  v71 = *(_QWORD *)(v76 + 88);
  v68 = (__int16 *)sub_1698280();
  sub_169D3F0((__int64)v90, 1.0);
  sub_169E320(&v92, v90, v68);
  sub_1698460((__int64)v90);
  sub_16A3360((__int64)v91, *(__int16 **)(v71 + 32), 0, (bool *)v90);
  v64 = v92;
  v66 = *(void **)(v71 + 32);
  v19 = sub_16982C0();
  v87 = 0;
  v20 = v64;
  v74 = v19;
  if ( v66 == v64 )
  {
    v28 = v71 + 32;
    if ( v19 == v64 )
      v29 = sub_169CB90(v28, (__int64)&v92);
    else
      v29 = sub_1698510(v28, (__int64)&v92);
    v20 = v92;
    v87 = v29;
  }
  if ( v74 == v20 )
  {
    v42 = v93;
    if ( v93 )
    {
      if ( v93 != v93 + 32LL * *(_QWORD *)(v93 - 8) )
      {
        v73 = v13;
        v43 = v93 + 32LL * *(_QWORD *)(v93 - 8);
        v44 = v93;
        do
        {
          v43 -= 32;
          sub_127D120((_QWORD *)(v43 + 8));
        }
        while ( v44 != v43 );
        v42 = v44;
        v11 = a2;
        v13 = v73;
      }
      j_j_j___libc_free_0_0(v42 - 8);
    }
  }
  else
  {
    sub_1698460((__int64)&v92);
  }
  if ( v87 )
  {
    v30 = *(__int64 **)v10[1];
    v31 = sub_1D309E0(
            v30,
            162,
            (__int64)v10[3],
            *(unsigned int *)v10[4],
            *((const void ***)v10[4] + 1),
            0,
            1.0,
            a8,
            *(double *)a9.m128i_i64,
            *(_OWORD *)(*(_QWORD *)(v11 + 32) + 40LL));
    *((_QWORD *)&v58 + 1) = v13;
    *(_QWORD *)&v58 = a3;
    return sub_1D3A900(
             v30,
             *(_DWORD *)v10[2],
             (__int64)v10[3],
             *(unsigned int *)v10[4],
             *((const void ***)v10[4] + 1),
             a5,
             (__m128)0x3FF0000000000000uLL,
             a8,
             a9,
             v31,
             v32,
             v58,
             a3,
             v13);
  }
  v77 = *(_QWORD *)(v76 + 88);
  sub_169D3F0((__int64)v90, -1.0);
  sub_169E320(&v92, v90, v68);
  sub_1698460((__int64)v90);
  sub_16A3360((__int64)v91, *(__int16 **)(v77 + 32), 0, (bool *)v90);
  v17 = v77;
  v21 = v92;
  if ( *(void **)(v77 + 32) == v92 )
  {
    v27 = v77 + 32;
    if ( v74 == v92 )
      v87 = sub_169CB90(v27, (__int64)&v92);
    else
      v87 = sub_1698510(v27, (__int64)&v92);
    v21 = v92;
  }
  if ( v74 == v21 )
  {
    v15 = v93;
    if ( v93 )
    {
      v39 = v93 + 32LL * *(_QWORD *)(v93 - 8);
      if ( v93 != v39 )
      {
        v85 = v13;
        v40 = v11;
        v41 = v93;
        do
        {
          v39 -= 32;
          sub_127D120((_QWORD *)(v39 + 8));
        }
        while ( v41 != v39 );
        v15 = v41;
        v11 = v40;
        v13 = v85;
      }
      j_j_j___libc_free_0_0(v15 - 8);
    }
  }
  else
  {
    sub_1698460((__int64)&v92);
  }
  if ( v87 )
  {
    v35 = *(__int64 **)v10[1];
    *((_QWORD *)&v62 + 1) = v13;
    *(_QWORD *)&v62 = a3;
    v83 = sub_1D309E0(
            v35,
            162,
            (__int64)v10[3],
            *(unsigned int *)v10[4],
            *((const void ***)v10[4] + 1),
            0,
            -1.0,
            a8,
            *(double *)a9.m128i_i64,
            v62);
    v84 = v36;
    v37 = sub_1D309E0(
            *(__int64 **)v10[1],
            162,
            (__int64)v10[3],
            *(unsigned int *)v10[4],
            *((const void ***)v10[4] + 1),
            0,
            -1.0,
            a8,
            *(double *)a9.m128i_i64,
            *(_OWORD *)(*(_QWORD *)(v11 + 32) + 40LL));
    *((_QWORD *)&v61 + 1) = v13;
    *(_QWORD *)&v61 = a3;
    return sub_1D3A900(
             v35,
             *(_DWORD *)v10[2],
             (__int64)v10[3],
             *(unsigned int *)v10[4],
             *((const void ***)v10[4] + 1),
             a5,
             (__m128)0xBFF0000000000000LL,
             a8,
             a9,
             v37,
             v38,
             v61,
             v83,
             v84);
  }
  else
  {
LABEL_15:
    v22 = sub_1D23470(
            *(_QWORD *)(*(_QWORD *)(v11 + 32) + 40LL),
            *(_QWORD *)(*(_QWORD *)(v11 + 32) + 48LL),
            v15,
            v16,
            v17,
            v18);
    if ( !v22 )
      return 0;
    v65 = v22;
    v78 = *(_QWORD *)(v22 + 88);
    v72 = (__int16 *)sub_1698280();
    sub_169D3F0((__int64)v90, 1.0);
    sub_169E320(&v92, v90, v72);
    sub_1698460((__int64)v90);
    sub_16A3360((__int64)v91, *(__int16 **)(v78 + 32), 0, (bool *)v90);
    v67 = v92;
    v69 = *(void **)(v78 + 32);
    v23 = sub_16982C0();
    v24 = v67;
    v88 = 0;
    v75 = v23;
    v25 = v65;
    if ( v69 == v67 )
    {
      v33 = v78 + 32;
      if ( v67 == v23 )
        v34 = sub_169CB90(v33, (__int64)&v92);
      else
        v34 = sub_1698510(v33, (__int64)&v92);
      v24 = v92;
      v25 = v65;
      v88 = v34;
    }
    if ( v75 == v24 )
    {
      v49 = v93;
      if ( v93 )
      {
        if ( v93 != v93 + 32LL * *(_QWORD *)(v93 - 8) )
        {
          v81 = v10;
          v50 = v93 + 32LL * *(_QWORD *)(v93 - 8);
          v70 = v13;
          v51 = v25;
          v52 = v11;
          v53 = v93;
          do
          {
            v50 -= 32;
            sub_127D120((_QWORD *)(v50 + 8));
          }
          while ( v53 != v50 );
          v49 = v53;
          v25 = v51;
          v11 = v52;
          v10 = v81;
          v13 = v70;
        }
        v82 = v25;
        j_j_j___libc_free_0_0(v49 - 8);
        v25 = v82;
      }
    }
    else
    {
      v79 = v25;
      sub_1698460((__int64)&v92);
      v25 = v79;
    }
    if ( !v88 )
    {
      v80 = *(_QWORD *)(v25 + 88);
      sub_169D3F0((__int64)v90, -1.0);
      sub_169E320(&v92, v90, v72);
      sub_1698460((__int64)v90);
      sub_16A3360((__int64)v91, *(__int16 **)(v80 + 32), 0, (bool *)v90);
      v26 = v92;
      if ( *(void **)(v80 + 32) == v92 )
      {
        v45 = v80 + 32;
        if ( v75 == v92 )
          v88 = sub_169CB90(v45, (__int64)&v92);
        else
          v88 = sub_1698510(v45, (__int64)&v92);
        v26 = v92;
      }
      if ( v26 == v75 )
      {
        v54 = v93;
        if ( v93 )
        {
          v55 = v93 + 32LL * *(_QWORD *)(v93 - 8);
          if ( v93 != v55 )
          {
            v86 = v13;
            v56 = v11;
            v57 = v93;
            do
            {
              v55 -= 32;
              sub_127D120((_QWORD *)(v55 + 8));
            }
            while ( v57 != v55 );
            v54 = v57;
            v11 = v56;
            v13 = v86;
          }
          j_j_j___libc_free_0_0(v54 - 8);
        }
      }
      else
      {
        sub_1698460((__int64)&v92);
      }
      if ( v88 )
      {
        *((_QWORD *)&v60 + 1) = v13;
        *(_QWORD *)&v60 = a3;
        return sub_1D3A900(
                 *(__int64 **)v10[1],
                 *(_DWORD *)v10[2],
                 (__int64)v10[3],
                 *(unsigned int *)v10[4],
                 *((const void ***)v10[4] + 1),
                 a5,
                 (__m128)0xBFF0000000000000LL,
                 a8,
                 a9,
                 **(_QWORD **)(v11 + 32),
                 *(__int16 **)(*(_QWORD *)(v11 + 32) + 8LL),
                 v60,
                 a3,
                 v13);
      }
      return 0;
    }
    v46 = *(__int64 **)v10[1];
    *((_QWORD *)&v63 + 1) = v13;
    *(_QWORD *)&v63 = a3;
    v47 = sub_1D309E0(
            v46,
            162,
            (__int64)v10[3],
            *(unsigned int *)v10[4],
            *((const void ***)v10[4] + 1),
            0,
            1.0,
            a8,
            *(double *)a9.m128i_i64,
            v63);
    *((_QWORD *)&v59 + 1) = v13;
    *(_QWORD *)&v59 = a3;
    return sub_1D3A900(
             v46,
             *(_DWORD *)v10[2],
             (__int64)v10[3],
             *(unsigned int *)v10[4],
             *((const void ***)v10[4] + 1),
             a5,
             (__m128)0x3FF0000000000000uLL,
             a8,
             a9,
             **(_QWORD **)(v11 + 32),
             *(__int16 **)(*(_QWORD *)(v11 + 32) + 8LL),
             v59,
             v47,
             v48);
  }
}
