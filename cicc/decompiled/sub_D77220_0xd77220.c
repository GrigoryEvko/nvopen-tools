// Function: sub_D77220
// Address: 0xd77220
//
__int64 *__fastcall sub_D77220(
        __int64 *a1,
        unsigned int a2,
        int a3,
        int a4,
        __int64 a5,
        __int64 a6,
        __int64 *a7,
        __int64 *a8,
        __int64 *a9,
        __int64 *a10,
        __int64 *a11,
        __int64 *a12,
        __int64 *a13,
        _QWORD *a14)
{
  __int64 v19; // r9
  __int64 v20; // r9
  __int64 v21; // r9
  __int64 v22; // r11
  __int64 v23; // r11
  __int64 v24; // r8
  __int64 v25; // r8
  __int64 v26; // r8
  __int64 v27; // r8
  __int64 v28; // r8
  __int64 v29; // r8
  __int64 v30; // rdi
  __int64 v31; // rdi
  __int64 v32; // rdi
  __int64 v33; // rsi
  __int64 v34; // rsi
  __int64 v35; // rsi
  __int64 v36; // rcx
  __int64 v37; // rcx
  __int64 v38; // rcx
  _QWORD *v39; // rdx
  _QWORD *v40; // rdx
  __int64 v41; // rdx
  __int64 v42; // rax
  __int64 v43; // r9
  _QWORD *v44; // rbx
  _QWORD *v45; // r14
  __int64 *v46; // r15
  __int64 *v47; // r13
  __int64 v48; // rdi
  __int64 v49; // r15
  __int64 v50; // r13
  __int64 v51; // rdi
  __int64 v52; // rbx
  __int64 v53; // r13
  __int64 v54; // rdi
  __int64 v55; // rdi
  __int64 v56; // r15
  __int64 v57; // r14
  __int64 v58; // rbx
  __int64 v59; // r13
  __int64 v60; // rdi
  __int64 v61; // rdi
  __int64 v62; // rdi
  __int64 v63; // rdi
  __int64 v64; // rbx
  __int64 v65; // r13
  __int64 v66; // rdi
  __int64 v67; // rbx
  __int64 v68; // r13
  __int64 v69; // rdi
  __int64 v72; // [rsp+8h] [rbp-138h]
  _QWORD *v73; // [rsp+10h] [rbp-130h] BYREF
  _QWORD *v74; // [rsp+18h] [rbp-128h]
  __int64 v75; // [rsp+20h] [rbp-120h]
  __int64 v76; // [rsp+30h] [rbp-110h] BYREF
  __int64 v77; // [rsp+38h] [rbp-108h]
  __int64 v78; // [rsp+40h] [rbp-100h]
  __int64 v79; // [rsp+50h] [rbp-F0h] BYREF
  __int64 v80; // [rsp+58h] [rbp-E8h]
  __int64 v81; // [rsp+60h] [rbp-E0h]
  __int64 v82; // [rsp+70h] [rbp-D0h] BYREF
  __int64 v83; // [rsp+78h] [rbp-C8h]
  __int64 v84; // [rsp+80h] [rbp-C0h]
  __int64 v85; // [rsp+90h] [rbp-B0h] BYREF
  __int64 v86; // [rsp+98h] [rbp-A8h]
  __int64 v87; // [rsp+A0h] [rbp-A0h]
  __int64 v88[2]; // [rsp+B0h] [rbp-90h] BYREF
  __int64 v89; // [rsp+C0h] [rbp-80h]
  __int64 v90[2]; // [rsp+D0h] [rbp-70h] BYREF
  __int64 v91; // [rsp+E0h] [rbp-60h]
  _QWORD v92[2]; // [rsp+F0h] [rbp-50h] BYREF
  __int64 v93; // [rsp+100h] [rbp-40h]

  v19 = *a7;
  *a7 = 0;
  v92[0] = v19;
  v20 = a7[1];
  a7[1] = 0;
  v92[1] = v20;
  v21 = a7[2];
  a7[2] = 0;
  v22 = *a8;
  v93 = v21;
  v90[0] = v22;
  v90[1] = a8[1];
  v23 = a8[2];
  a8[2] = 0;
  v91 = v23;
  a8[1] = 0;
  *a8 = 0;
  v24 = *a9;
  *a9 = 0;
  v88[0] = v24;
  v25 = a9[1];
  a9[1] = 0;
  v88[1] = v25;
  v26 = a9[2];
  a9[2] = 0;
  v89 = v26;
  v27 = *a10;
  *a10 = 0;
  v85 = v27;
  v28 = a10[1];
  a10[1] = 0;
  v86 = v28;
  v29 = a10[2];
  a10[2] = 0;
  v30 = *a11;
  *a11 = 0;
  v82 = v30;
  v31 = a11[1];
  a11[1] = 0;
  v83 = v31;
  v32 = a11[2];
  a11[2] = 0;
  v33 = *a12;
  v84 = v32;
  v79 = v33;
  v34 = a12[1];
  v87 = v29;
  v80 = v34;
  v35 = a12[2];
  a12[2] = 0;
  a12[1] = 0;
  *a12 = 0;
  v36 = *a13;
  *a13 = 0;
  v76 = v36;
  v37 = a13[1];
  a13[1] = 0;
  v77 = v37;
  v38 = a13[2];
  a13[2] = 0;
  v39 = (_QWORD *)*a14;
  *a14 = 0;
  v73 = v39;
  v40 = (_QWORD *)a14[1];
  a14[1] = 0;
  v74 = v40;
  v41 = a14[2];
  a14[2] = 0;
  v81 = v35;
  v78 = v38;
  v75 = v41;
  v42 = sub_22077B0(112);
  if ( v42 )
  {
    v43 = a6;
    v35 = a2;
    v72 = v42;
    sub_9C6E00(v42, a2, a3, a4, a5, v43, v92, v90, v88, &v85, &v82, &v79, &v76, (__int64)&v73);
    v42 = v72;
  }
  v44 = v74;
  v45 = v73;
  *a1 = v42;
  if ( v44 != v45 )
  {
    do
    {
      v46 = (__int64 *)v45[12];
      v47 = (__int64 *)v45[11];
      if ( v46 != v47 )
      {
        do
        {
          v48 = *v47;
          if ( *v47 )
          {
            v35 = v47[2] - v48;
            j_j___libc_free_0(v48, v35);
          }
          v47 += 3;
        }
        while ( v46 != v47 );
        v47 = (__int64 *)v45[11];
      }
      if ( v47 )
      {
        v35 = v45[13] - (_QWORD)v47;
        j_j___libc_free_0(v47, v35);
      }
      v49 = v45[9];
      v50 = v45[8];
      if ( v49 != v50 )
      {
        do
        {
          v51 = *(_QWORD *)(v50 + 8);
          if ( v51 != v50 + 24 )
            _libc_free(v51, v35);
          v50 += 72;
        }
        while ( v49 != v50 );
        v50 = v45[8];
      }
      if ( v50 )
      {
        v35 = v45[10] - v50;
        j_j___libc_free_0(v50, v35);
      }
      if ( (_QWORD *)*v45 != v45 + 3 )
        _libc_free(*v45, v35);
      v45 += 14;
    }
    while ( v44 != v45 );
    v45 = v73;
  }
  if ( v45 )
  {
    v35 = v75 - (_QWORD)v45;
    j_j___libc_free_0(v45, v75 - (_QWORD)v45);
  }
  v52 = v77;
  v53 = v76;
  if ( v77 != v76 )
  {
    do
    {
      v54 = *(_QWORD *)(v53 + 72);
      if ( v54 != v53 + 88 )
        _libc_free(v54, v35);
      v55 = *(_QWORD *)(v53 + 8);
      if ( v55 != v53 + 24 )
        _libc_free(v55, v35);
      v53 += 136;
    }
    while ( v52 != v53 );
    v53 = v76;
  }
  if ( v53 )
    j_j___libc_free_0(v53, v78 - v53);
  v56 = v80;
  v57 = v79;
  if ( v80 != v79 )
  {
    do
    {
      v58 = *(_QWORD *)(v57 + 48);
      v59 = *(_QWORD *)(v57 + 40);
      if ( v58 != v59 )
      {
        do
        {
          if ( *(_DWORD *)(v59 + 40) > 0x40u )
          {
            v60 = *(_QWORD *)(v59 + 32);
            if ( v60 )
              j_j___libc_free_0_0(v60);
          }
          if ( *(_DWORD *)(v59 + 24) > 0x40u )
          {
            v61 = *(_QWORD *)(v59 + 16);
            if ( v61 )
              j_j___libc_free_0_0(v61);
          }
          v59 += 48;
        }
        while ( v58 != v59 );
        v59 = *(_QWORD *)(v57 + 40);
      }
      if ( v59 )
        j_j___libc_free_0(v59, *(_QWORD *)(v57 + 56) - v59);
      if ( *(_DWORD *)(v57 + 32) > 0x40u )
      {
        v62 = *(_QWORD *)(v57 + 24);
        if ( v62 )
          j_j___libc_free_0_0(v62);
      }
      if ( *(_DWORD *)(v57 + 16) > 0x40u )
      {
        v63 = *(_QWORD *)(v57 + 8);
        if ( v63 )
          j_j___libc_free_0_0(v63);
      }
      v57 += 64;
    }
    while ( v56 != v57 );
    v57 = v79;
  }
  if ( v57 )
    j_j___libc_free_0(v57, v81 - v57);
  v64 = v83;
  v65 = v82;
  if ( v83 != v82 )
  {
    do
    {
      v66 = *(_QWORD *)(v65 + 16);
      if ( v66 )
        j_j___libc_free_0(v66, *(_QWORD *)(v65 + 32) - v66);
      v65 += 40;
    }
    while ( v64 != v65 );
    v65 = v82;
  }
  if ( v65 )
    j_j___libc_free_0(v65, v84 - v65);
  v67 = v86;
  v68 = v85;
  if ( v86 != v85 )
  {
    do
    {
      v69 = *(_QWORD *)(v68 + 16);
      if ( v69 )
        j_j___libc_free_0(v69, *(_QWORD *)(v68 + 32) - v69);
      v68 += 40;
    }
    while ( v67 != v68 );
    v68 = v85;
  }
  if ( v68 )
    j_j___libc_free_0(v68, v87 - v68);
  if ( v88[0] )
    j_j___libc_free_0(v88[0], v89 - v88[0]);
  if ( v90[0] )
    j_j___libc_free_0(v90[0], v91 - v90[0]);
  if ( v92[0] )
    j_j___libc_free_0(v92[0], v93 - v92[0]);
  return a1;
}
