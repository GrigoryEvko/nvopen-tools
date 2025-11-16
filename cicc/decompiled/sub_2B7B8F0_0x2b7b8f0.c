// Function: sub_2B7B8F0
// Address: 0x2b7b8f0
//
__int64 ***__fastcall sub_2B7B8F0(
        __int64 a1,
        char *a2,
        unsigned __int64 a3,
        __int64 a4,
        __int64 a5,
        __int64 a6,
        const void *a7,
        __int64 a8,
        void (__fastcall *a9)(__int64, __int64 *, __int64),
        __int64 a10)
{
  __int64 *v13; // rsi
  __int64 v14; // rax
  __int64 v15; // rcx
  int *v16; // rdx
  __int64 v17; // rdi
  __int64 v18; // r11
  __int64 v19; // rax
  void *v20; // r15
  __int64 v21; // r9
  __int64 v22; // r8
  __int64 ***v23; // rax
  __int64 v24; // r8
  __int64 v25; // r9
  __int64 v26; // rcx
  __int64 v27; // rsi
  __int64 v28; // rax
  _DWORD *v29; // rdx
  __int64 v30; // rax
  __int64 *v31; // rsi
  int *v32; // rdx
  __int64 v33; // rcx
  __int64 v34; // r10
  __int64 v35; // rax
  __int64 v36; // rdi
  __int64 v37; // r9
  void *v38; // r11
  __int64 v39; // r8
  __int64 ***v40; // r14
  __int64 v41; // rcx
  __int64 v42; // rsi
  __int64 v43; // rax
  _DWORD *v44; // rdx
  _QWORD *v45; // r14
  __int64 v46; // rcx
  __int64 v47; // rdx
  __int64 v48; // rcx
  __int64 v49; // r8
  __int64 v50; // r9
  __int64 v51; // rax
  __int64 *v52; // rdx
  signed __int64 v54; // r12
  unsigned int v55; // r15d
  __int64 v56; // rax
  int *v57; // r10
  __int64 v58; // r9
  __int64 v59; // rsi
  __int64 v60; // rdi
  __int64 v61; // rax
  __int64 v62; // r11
  __int64 v63; // rsi
  unsigned __int64 v64; // rsi
  _DWORD *v65; // rdi
  __int64 v66; // r9
  _DWORD *v67; // rax
  _DWORD *v68; // rcx
  _DWORD *i; // rsi
  _QWORD *v70; // rax
  _QWORD *v71; // rdi
  __int64 v72; // rax
  __int64 v73; // r9
  __int64 v74; // rdx
  __int64 ***v75; // rax
  __int64 v76; // rcx
  __int64 *v77; // rdi
  __int64 v78; // rsi
  __int64 v79; // rax
  _DWORD *v80; // rcx
  int *v81; // rdx
  __int64 v82; // rax
  __int64 v83; // rdx
  __int64 v84; // rax
  __int64 v85; // r9
  __int64 v86; // rsi
  __int64 v87; // rsi
  __int64 ***v88; // rax
  __int64 v89; // r12
  __int64 v90; // r11
  unsigned int v91; // [rsp+Ch] [rbp-D4h]
  unsigned int v92; // [rsp+18h] [rbp-C8h]
  __int64 v93; // [rsp+28h] [rbp-B8h]
  __int64 v94; // [rsp+30h] [rbp-B0h] BYREF
  __int64 v95; // [rsp+38h] [rbp-A8h]
  __int64 v96[2]; // [rsp+40h] [rbp-A0h] BYREF
  __int64 v97; // [rsp+50h] [rbp-90h] BYREF
  __int64 v98; // [rsp+58h] [rbp-88h]
  __int64 v99; // [rsp+60h] [rbp-80h]
  __int64 v100; // [rsp+68h] [rbp-78h]
  void *dest; // [rsp+70h] [rbp-70h] BYREF
  __int64 v102; // [rsp+78h] [rbp-68h]
  __int64 v103; // [rsp+80h] [rbp-60h] BYREF
  __int64 v104; // [rsp+88h] [rbp-58h]

  v94 = a4;
  v95 = a5;
  *(_BYTE *)(a1 + 8) = 1;
  if ( a9 )
  {
    v13 = *(__int64 **)(a1 + 80);
    v14 = *(_QWORD *)(a1 + 120);
    v15 = *(unsigned int *)(a1 + 24);
    v16 = *(int **)(a1 + 16);
    v92 = a6;
    v17 = *v13;
    v18 = v14 + 3160;
    v19 = v14 + 3112;
    v20 = *(void **)(a1 + 112);
    v21 = *(_QWORD *)a1;
    v96[0] = *v13;
    v22 = *(_QWORD *)(v19 + 232);
    if ( *(_DWORD *)(a1 + 88) == 2 )
    {
      v63 = v13[1];
      dest = v20;
      v104 = v22;
      v102 = v19;
      v103 = v18;
      v23 = sub_2B7A630(v17, v63, v16, v15, (__int64)&dest, v21);
      --*(_DWORD *)(a1 + 88);
    }
    else
    {
      dest = v20;
      v104 = v22;
      v102 = v19;
      v103 = v18;
      v23 = sub_2B7A630(v17, 0, v16, v15, (__int64)&dest, v21);
    }
    v96[0] = (__int64)v23;
    v26 = *(unsigned int *)(a1 + 24);
    v27 = *(_QWORD *)(a1 + 16);
    v28 = 0;
    if ( (_DWORD)v26 )
    {
      do
      {
        v29 = (_DWORD *)(v27 + 4LL * (unsigned int)v28);
        if ( *v29 != -1 )
          *v29 = v28;
        ++v28;
      }
      while ( v26 != v28 );
    }
    if ( v92 > *(_DWORD *)(*(_QWORD *)(v96[0] + 8) + 32LL) )
    {
      v91 = *(_DWORD *)(*(_QWORD *)(v96[0] + 8) + 32LL);
      dest = &v103;
      v102 = 0xC00000000LL;
      sub_11B1960((__int64)&dest, v92, -1, v26, v24, v25);
      v80 = dest;
      v81 = (int *)((char *)dest + 4 * v91);
      if ( dest != v81 )
      {
        v82 = 0;
        do
        {
          v83 = v82;
          v80[v82] = v82;
          ++v82;
        }
        while ( (4 * (unsigned __int64)v91 - 4) >> 2 != v83 );
        v81 = (int *)dest;
      }
      v84 = *(_QWORD *)(a1 + 120);
      v85 = *(_QWORD *)a1;
      v86 = *(_QWORD *)(v84 + 3344);
      v97 = *(_QWORD *)(a1 + 112);
      v98 = v84 + 3112;
      v100 = v86;
      v99 = v84 + 3160;
      v96[0] = (__int64)sub_2B7A630(v96[0], 0, v81, (unsigned int)v102, (__int64)&v97, v85);
      if ( dest != &v103 )
        _libc_free((unsigned __int64)dest);
    }
    a9(a10, v96, a1 + 16);
    **(_QWORD **)(a1 + 80) = v96[0];
  }
  if ( v95 )
  {
    v30 = *(_QWORD *)(a1 + 120);
    v31 = *(__int64 **)(a1 + 80);
    v32 = *(int **)(a1 + 16);
    v33 = *(unsigned int *)(a1 + 24);
    v34 = v30 + 3160;
    v35 = v30 + 3112;
    v36 = *v31;
    v37 = *(_QWORD *)a1;
    v38 = *(void **)(a1 + 112);
    if ( *(_DWORD *)(a1 + 88) == 2 )
    {
      v87 = v31[1];
      v104 = *(_QWORD *)(v35 + 232);
      dest = v38;
      v102 = v35;
      v103 = v34;
      v88 = sub_2B7A630(v36, v87, v32, v33, (__int64)&dest, v37);
      --*(_DWORD *)(a1 + 88);
      v40 = v88;
    }
    else
    {
      v104 = *(_QWORD *)(v35 + 232);
      dest = v38;
      v102 = v35;
      v103 = v34;
      v40 = sub_2B7A630(v36, 0, v32, v33, (__int64)&dest, v37);
    }
    v41 = *(unsigned int *)(a1 + 24);
    v42 = *(_QWORD *)(a1 + 16);
    v43 = 0;
    if ( (_DWORD)v41 )
    {
      do
      {
        v44 = (_DWORD *)(v42 + 4LL * (unsigned int)v43);
        if ( *v44 != -1 )
          *v44 = v43;
        ++v43;
      }
      while ( v41 != v43 );
    }
    v96[1] = a1;
    v96[0] = (__int64)&v94;
    if ( a8 )
    {
      v64 = *(unsigned int *)(a1 + 24);
      dest = &v103;
      v102 = 0xC00000000LL;
      sub_11B1960((__int64)&dest, v64, -1, v41, v39, a1 + 16);
      v65 = dest;
      v66 = a1 + 16;
      if ( 4 * a8 )
      {
        memmove(dest, a7, 4 * a8);
        v65 = dest;
        v66 = a1 + 16;
      }
      v67 = *(_DWORD **)(a1 + 16);
      v68 = &v67[*(unsigned int *)(a1 + 24)];
      for ( i = &v65[(unsigned int)v102]; v67 != v68; ++v65 )
      {
        if ( v65 == i )
          break;
        if ( *v67 != -1 )
          *v65 = *(_DWORD *)(a1 + 24) + *v67;
        ++v67;
      }
      v93 = v66;
      v70 = (_QWORD *)sub_ACADE0(v40[1]);
      v71 = sub_2B33530((__int64)v96, v70, v93);
      v72 = *(_QWORD *)(a1 + 120);
      v73 = *(_QWORD *)a1;
      v74 = *(_QWORD *)(v72 + 3344);
      v97 = *(_QWORD *)(a1 + 112);
      v98 = v72 + 3112;
      v100 = v74;
      v99 = v72 + 3160;
      v75 = sub_2B7A630((__int64)v71, (__int64)v40, (int *)dest, (unsigned int)v102, (__int64)&v97, v73);
      v76 = *(unsigned int *)(a1 + 24);
      v77 = (__int64 *)dest;
      v45 = v75;
      v78 = *(_QWORD *)(a1 + 16);
      v79 = 0;
      if ( (_DWORD)v76 )
      {
        do
        {
          if ( *((_DWORD *)v77 + (unsigned int)v79) != -1 )
            *(_DWORD *)(v78 + 4LL * (unsigned int)v79) = v79;
          ++v79;
        }
        while ( v76 != v79 );
        v77 = (__int64 *)dest;
      }
      if ( v77 != &v103 )
        _libc_free((unsigned __int64)v77);
    }
    else
    {
      v45 = sub_2B33530((__int64)v96, v40, a1 + 16);
    }
    **(_QWORD **)(a1 + 80) = v45;
  }
  v46 = *(unsigned int *)(a1 + 24);
  if ( a3 )
  {
    if ( !(_DWORD)v46 )
    {
      v54 = 4 * a3;
      if ( v54 >> 2 > (unsigned __int64)*(unsigned int *)(a1 + 28) )
      {
        sub_C8D5F0(a1 + 16, (const void *)(a1 + 32), v54 >> 2, 4u, a5, a6);
        v46 = *(unsigned int *)(a1 + 24);
      }
      if ( v54 )
      {
        memcpy((void *)(*(_QWORD *)(a1 + 16) + 4 * v46), a2, v54);
        LODWORD(v46) = *(_DWORD *)(a1 + 24);
      }
      v55 = v46 + (v54 >> 2);
      v52 = *(__int64 **)(a1 + 80);
      v46 = v55;
      *(_DWORD *)(a1 + 24) = v55;
      if ( !v55 )
        return (__int64 ***)*v52;
      goto LABEL_37;
    }
    dest = &v103;
    v102 = 0xC00000000LL;
    sub_11B1960((__int64)&dest, a3, -1, v46, a5, a6);
    if ( (int)a3 > 0 )
    {
      v51 = 0;
      do
      {
        v47 = *(int *)&a2[v51];
        if ( (_DWORD)v47 != -1 )
        {
          v48 = *(unsigned int *)(*(_QWORD *)(a1 + 16) + 4 * v47);
          v47 = (__int64)dest;
          *(_DWORD *)((char *)dest + v51) = v48;
        }
        v51 += 4;
      }
      while ( 4LL * (unsigned int)(a3 - 1) + 4 != v51 );
    }
    sub_2B310D0(a1 + 16, (__int64)&dest, v47, v48, v49, v50);
    if ( dest != &v103 )
      _libc_free((unsigned __int64)dest);
    v46 = *(unsigned int *)(a1 + 24);
  }
  v52 = *(__int64 **)(a1 + 80);
  if ( !(_DWORD)v46 )
    return (__int64 ***)*v52;
LABEL_37:
  v56 = *(_QWORD *)(a1 + 120);
  v57 = *(int **)(a1 + 16);
  v58 = *(_QWORD *)a1;
  v59 = *(_QWORD *)(v56 + 3344);
  v60 = v56 + 3160;
  v61 = v56 + 3112;
  if ( *(_DWORD *)(a1 + 88) == 2 )
  {
    v89 = v52[1];
    v90 = *v52;
    dest = *(void **)(a1 + 112);
    v103 = v60;
    v104 = v59;
    v102 = v61;
    return sub_2B7A630(v90, v89, v57, v46, (__int64)&dest, v58);
  }
  else
  {
    v62 = *v52;
    dest = *(void **)(a1 + 112);
    v103 = v60;
    v104 = v59;
    v102 = v61;
    return sub_2B7A630(v62, 0, v57, v46, (__int64)&dest, v58);
  }
}
