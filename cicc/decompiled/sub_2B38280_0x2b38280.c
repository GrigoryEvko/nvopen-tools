// Function: sub_2B38280
// Address: 0x2b38280
//
unsigned __int64 __fastcall sub_2B38280(
        __int64 a1,
        _BYTE *a2,
        unsigned __int64 a3,
        __int64 a4,
        __int64 a5,
        __int64 a6,
        const void *a7,
        __int64 a8,
        void (__fastcall *a9)(__int64, void **, __int64),
        __int64 a10)
{
  _DWORD *v11; // r12
  __int64 v12; // rbx
  __int64 *v13; // rsi
  const void *v14; // rcx
  __int64 v15; // r8
  signed __int64 v16; // rax
  __int64 v17; // r11
  int v18; // edx
  unsigned __int64 v19; // rax
  __int64 v20; // rcx
  __int64 v21; // rdi
  __int64 v22; // rax
  _DWORD *v23; // rdx
  __int64 *v24; // rsi
  const void *v25; // rcx
  __int64 v26; // r8
  signed __int64 v27; // rax
  int v28; // edx
  unsigned __int64 v29; // rax
  __int64 v30; // rsi
  __int64 v31; // rax
  _DWORD *v32; // rdx
  int v33; // esi
  __int64 *v34; // r8
  __int64 v35; // rax
  __int64 v36; // rdi
  int v37; // esi
  __int64 v38; // r13
  int v39; // eax
  __int64 v40; // rax
  __int64 v41; // rax
  int v42; // edx
  unsigned __int64 v43; // rax
  int v44; // edx
  __int64 v45; // r15
  unsigned int v46; // ebx
  __int64 v47; // rax
  __int64 v48; // rcx
  __int64 v49; // rax
  unsigned int v50; // edx
  __int64 *v51; // r13
  __int64 v52; // rsi
  __int64 v53; // rax
  char v54; // cl
  void *v55; // rax
  __int64 v56; // rdx
  __int64 v57; // rdx
  unsigned int v58; // eax
  int v59; // esi
  __int64 *v60; // r8
  unsigned int v61; // r13d
  int v62; // eax
  __int64 v63; // rax
  int v64; // esi
  __int64 v65; // rcx
  __int64 v66; // rdi
  int v67; // eax
  __int64 v68; // rax
  __int64 v69; // rax
  int v70; // edx
  unsigned __int64 v71; // rax
  int v72; // eax
  __int64 v73; // rdx
  __int64 v74; // rcx
  __int64 v75; // r8
  __int64 v76; // r9
  __int64 v77; // rax
  __int64 v78; // r8
  unsigned __int64 result; // rax
  const void *v80; // rcx
  __int64 v81; // rsi
  signed __int64 v82; // rcx
  signed __int64 v83; // rax
  int v84; // edx
  unsigned __int64 v85; // rax
  int v86; // r11d
  int v87; // eax
  unsigned int *v88; // rdx
  unsigned int *i; // rax
  int v90; // edx
  unsigned __int64 v91; // rsi
  int *v92; // rcx
  _DWORD *v93; // rdx
  unsigned __int64 v94; // r15
  _DWORD *v95; // r8
  int v96; // esi
  int *v97; // rdi
  __int64 v98; // rax
  __int64 v99; // rax
  int v100; // edx
  unsigned __int64 v101; // rax
  _BYTE *v102; // rdi
  bool v103; // cc
  __int64 v104; // [rsp-8h] [rbp-B8h]
  unsigned int v107; // [rsp+10h] [rbp-A0h]
  __int64 v108; // [rsp+18h] [rbp-98h]
  __int64 v109; // [rsp+18h] [rbp-98h]
  _DWORD *src; // [rsp+20h] [rbp-90h]
  char v111; // [rsp+28h] [rbp-88h]
  __int64 *v112; // [rsp+28h] [rbp-88h]
  int *v113; // [rsp+28h] [rbp-88h]
  void *dest; // [rsp+40h] [rbp-70h] BYREF
  __int64 v115; // [rsp+48h] [rbp-68h]
  _BYTE v116[96]; // [rsp+50h] [rbp-60h] BYREF

  v11 = (_DWORD *)a4;
  v12 = a5;
  *(_BYTE *)(a1 + 8) = 1;
  if ( a9 )
  {
    v13 = *(__int64 **)(a1 + 80);
    v14 = *(const void **)(a1 + 16);
    v15 = *(unsigned int *)(a1 + 24);
    if ( *(_DWORD *)(a1 + 88) == 2 )
    {
      v83 = sub_2B35AF0(a1, v13, (__int64)(v13 + 1), v14, v15, a6);
      v17 = a10;
      if ( v84 == 1 )
        *(_DWORD *)(a1 + 128) = 1;
      if ( __OFADD__(*(_QWORD *)(a1 + 120), v83) )
      {
        v103 = v83 <= 0;
        v85 = 0x7FFFFFFFFFFFFFFFLL;
        if ( v103 )
          v85 = 0x8000000000000000LL;
      }
      else
      {
        v85 = *(_QWORD *)(a1 + 120) + v83;
      }
      *(_QWORD *)(a1 + 120) = v85;
    }
    else
    {
      dest = 0;
      v16 = sub_2B35AF0(a1, v13, (__int64)&dest, v14, v15, a6);
      v17 = a10;
      if ( v18 == 1 )
        *(_DWORD *)(a1 + 128) = 1;
      if ( __OFADD__(*(_QWORD *)(a1 + 120), v16) )
      {
        v103 = v16 <= 0;
        v19 = 0x7FFFFFFFFFFFFFFFLL;
        if ( v103 )
          v19 = 0x8000000000000000LL;
      }
      else
      {
        v19 = *(_QWORD *)(a1 + 120) + v16;
      }
      *(_QWORD *)(a1 + 120) = v19;
    }
    v20 = *(unsigned int *)(a1 + 24);
    v21 = *(_QWORD *)(a1 + 16);
    v22 = 0;
    if ( (_DWORD)v20 )
    {
      do
      {
        v23 = (_DWORD *)(v21 + 4LL * (unsigned int)v22);
        if ( *v23 != -1 )
          *v23 = v22;
        ++v22;
      }
      while ( v20 != v22 );
    }
    dest = (void *)(*v13 & 0xFFFFFFFFFFFFFFF8LL);
    a9(v17, &dest, a1 + 16);
    **(_QWORD **)(a1 + 80) = (unsigned __int64)dest & 0xFFFFFFFFFFFFFFFBLL;
  }
  if ( v12 )
  {
    v24 = *(__int64 **)(a1 + 80);
    v25 = *(const void **)(a1 + 16);
    v26 = *(unsigned int *)(a1 + 24);
    if ( *(_DWORD *)(a1 + 88) == 2 )
    {
      v27 = sub_2B35AF0(a1, v24, (__int64)(v24 + 1), v25, v26, a6);
      if ( v90 != 1 )
        goto LABEL_16;
    }
    else
    {
      dest = 0;
      v27 = sub_2B35AF0(a1, v24, (__int64)&dest, v25, v26, a6);
      if ( v28 != 1 )
        goto LABEL_16;
    }
    *(_DWORD *)(a1 + 128) = 1;
LABEL_16:
    if ( __OFADD__(*(_QWORD *)(a1 + 120), v27) )
    {
      v103 = v27 <= 0;
      v29 = 0x7FFFFFFFFFFFFFFFLL;
      if ( v103 )
        v29 = 0x8000000000000000LL;
    }
    else
    {
      v29 = *(_QWORD *)(a1 + 120) + v27;
    }
    a4 = *(unsigned int *)(a1 + 24);
    v30 = *(_QWORD *)(a1 + 16);
    *(_QWORD *)(a1 + 120) = v29;
    v31 = 0;
    if ( (_DWORD)a4 )
    {
      do
      {
        v32 = (_DWORD *)(v30 + 4LL * (unsigned int)v31);
        if ( *v32 != -1 )
          *v32 = v31;
        ++v31;
      }
      while ( a4 != v31 );
    }
    if ( a8 )
    {
      v91 = *(unsigned int *)(a1 + 24);
      dest = v116;
      v115 = 0xC00000000LL;
      sub_11B1960((__int64)&dest, v91, -1, a4, a5, a6);
      v92 = (int *)dest;
      if ( 4 * a8 )
      {
        memmove(dest, a7, 4 * a8);
        v92 = (int *)dest;
      }
      v93 = *(_DWORD **)(a1 + 16);
      v94 = (unsigned int)v115;
      v95 = &v93[*(unsigned int *)(a1 + 24)];
      v96 = *(_DWORD *)(a1 + 24);
      v97 = &v92[(unsigned int)v115];
      if ( v93 != v95 && v92 != v97 )
      {
        do
        {
          if ( *v93 != -1 )
          {
            *v92 = v96 + *v93;
            v96 = *(_DWORD *)(a1 + 24);
          }
          ++v92;
          ++v93;
        }
        while ( v92 != v97 && v93 != v95 );
        v92 = (int *)dest;
        v94 = (unsigned int)v115;
      }
      v113 = v92;
      v98 = sub_2B08680(*(_QWORD *)a1, v96);
      v99 = sub_2B097B0(*(__int64 **)(a1 + 112), 6, v98, v113, v94, 0, 0, 0);
      a5 = v104;
      if ( v100 == 1 )
        *(_DWORD *)(a1 + 128) = 1;
      if ( __OFADD__(*(_QWORD *)(a1 + 120), v99) )
      {
        v103 = v99 <= 0;
        v101 = 0x7FFFFFFFFFFFFFFFLL;
        if ( v103 )
          v101 = 0x8000000000000000LL;
      }
      else
      {
        v101 = *(_QWORD *)(a1 + 120) + v99;
      }
      v102 = dest;
      *(_QWORD *)(a1 + 120) = v101;
      if ( v102 != v116 )
        _libc_free((unsigned __int64)v102);
    }
    src = &v11[4 * v12];
    if ( v11 == src )
      goto LABEL_63;
    do
    {
      v45 = *(_QWORD *)v11;
      v46 = v11[2];
      v34 = *(__int64 **)(***(_QWORD ***)v11 + 8LL);
      v47 = *(_QWORD *)(a1 + 184);
      v48 = *(_QWORD *)(v47 + 3528);
      v49 = *(unsigned int *)(v47 + 3544);
      if ( (_DWORD)v49 )
      {
        v50 = (v49 - 1) & (((unsigned int)v45 >> 9) ^ ((unsigned int)v45 >> 4));
        v51 = (__int64 *)(v48 + 24LL * v50);
        v52 = *v51;
        if ( v45 == *v51 )
        {
LABEL_38:
          if ( v51 != (__int64 *)(v48 + 24 * v49) )
          {
            v53 = sub_BCCE00((_QWORD *)*v34, *((_DWORD *)v51 + 2));
            v54 = *((_BYTE *)v51 + 16);
            v34 = (__int64 *)v53;
            goto LABEL_40;
          }
        }
        else
        {
          v86 = 1;
          while ( v52 != -4096 )
          {
            v50 = (v49 - 1) & (v86 + v50);
            v51 = (__int64 *)(v48 + 24LL * v50);
            v52 = *v51;
            if ( v45 == *v51 )
              goto LABEL_38;
            ++v86;
          }
        }
      }
      v54 = 1;
LABEL_40:
      v111 = v54;
      if ( *(__int64 **)a1 == v34 )
        goto LABEL_58;
      v108 = (__int64)v34;
      v55 = (void *)sub_9208B0(*(_QWORD *)(*(_QWORD *)(a1 + 184) + 3344LL), *(_QWORD *)a1);
      v115 = v56;
      dest = v55;
      v107 = sub_CA1930(&dest);
      dest = (void *)sub_9208B0(*(_QWORD *)(*(_QWORD *)(a1 + 184) + 3344LL), v108);
      v115 = v57;
      v58 = sub_CA1930(&dest);
      v59 = *(_DWORD *)(v45 + 120);
      v60 = (__int64 *)v108;
      v61 = 39 - ((v111 == 0) - 1);
      if ( v107 <= v58 )
        v61 = 38;
      v112 = *(__int64 **)(a1 + 112);
      if ( !v59 )
        v59 = *(_DWORD *)(v45 + 8);
      v62 = *(unsigned __int8 *)(v108 + 8);
      if ( (_BYTE)v62 == 17 )
      {
        v59 *= *(_DWORD *)(v108 + 32);
LABEL_47:
        v60 = **(__int64 ***)(v108 + 16);
        goto LABEL_48;
      }
      if ( (unsigned int)(v62 - 17) <= 1 )
        goto LABEL_47;
LABEL_48:
      v63 = sub_BCDA70(v60, v59);
      v64 = *(_DWORD *)(v45 + 120);
      v65 = v63;
      if ( !v64 )
        v64 = *(_DWORD *)(v45 + 8);
      v66 = *(_QWORD *)a1;
      v67 = *(unsigned __int8 *)(*(_QWORD *)a1 + 8LL);
      if ( (_BYTE)v67 == 17 )
      {
        v64 *= *(_DWORD *)(v66 + 32);
      }
      else if ( (unsigned int)(v67 - 17) > 1 )
      {
        goto LABEL_53;
      }
      v66 = **(_QWORD **)(v66 + 16);
LABEL_53:
      v109 = v65;
      v68 = sub_BCDA70((__int64 *)v66, v64);
      v69 = sub_DFD060(v112, v61, v68, v109);
      if ( v70 == 1 )
        *(_DWORD *)(a1 + 128) = 1;
      if ( __OFADD__(*(_QWORD *)(a1 + 120), v69) )
      {
        v103 = v69 <= 0;
        v71 = 0x8000000000000000LL;
        if ( !v103 )
          v71 = 0x7FFFFFFFFFFFFFFFLL;
      }
      else
      {
        v71 = *(_QWORD *)(a1 + 120) + v69;
      }
      *(_QWORD *)(a1 + 120) = v71;
      v34 = *(__int64 **)a1;
LABEL_58:
      v33 = *(_DWORD *)(v45 + 120);
      if ( !v33 )
        v33 = *(_DWORD *)(v45 + 8);
      v72 = *((unsigned __int8 *)v34 + 8);
      if ( (_BYTE)v72 == 17 )
      {
        v33 *= *((_DWORD *)v34 + 8);
LABEL_26:
        v34 = *(__int64 **)v34[2];
        goto LABEL_27;
      }
      if ( (unsigned int)(v72 - 17) <= 1 )
        goto LABEL_26;
LABEL_27:
      v35 = sub_BCDA70(v34, v33);
      v36 = *(_QWORD *)a1;
      v37 = *(_DWORD *)(a1 + 24);
      v38 = v35;
      v39 = *(unsigned __int8 *)(*(_QWORD *)a1 + 8LL);
      if ( (_BYTE)v39 == 17 )
      {
        v37 *= *(_DWORD *)(v36 + 32);
      }
      else if ( (unsigned int)(v39 - 17) > 1 )
      {
        goto LABEL_30;
      }
      v36 = **(_QWORD **)(v36 + 16);
LABEL_30:
      v40 = sub_BCDA70((__int64 *)v36, v37);
      v41 = sub_DFBC30(*(__int64 **)(a1 + 112), 4, v40, 0, 0, 0, v46, v38, 0, 0, 0);
      if ( v42 == 1 )
        *(_DWORD *)(a1 + 128) = 1;
      if ( __OFADD__(*(_QWORD *)(a1 + 120), v41) )
      {
        a4 = 0x7FFFFFFFFFFFFFFFLL;
        v103 = v41 <= 0;
        v43 = 0x8000000000000000LL;
        if ( !v103 )
          v43 = 0x7FFFFFFFFFFFFFFFLL;
      }
      else
      {
        v43 = *(_QWORD *)(a1 + 120) + v41;
      }
      v44 = *(_DWORD *)(a1 + 24);
      *(_QWORD *)(a1 + 120) = v43;
      if ( v44 )
      {
        v87 = *(_DWORD *)(v45 + 120);
        if ( !v87 )
          v87 = *(_DWORD *)(v45 + 8);
        a4 = *(_QWORD *)(a1 + 16);
        v88 = (unsigned int *)(a4 + 4LL * (v46 + v87));
        for ( i = (unsigned int *)(a4 + 4LL * v46); i != v88; ++v46 )
          *i++ = v46;
      }
      v11 += 4;
    }
    while ( src != v11 );
  }
LABEL_63:
  if ( a3 )
  {
    if ( *(_DWORD *)(a1 + 24) )
    {
      dest = v116;
      v115 = 0xC00000000LL;
      sub_11B1960((__int64)&dest, a3, -1, a4, a5, a6);
      if ( (int)a3 > 0 )
      {
        v77 = 0;
        do
        {
          v73 = *(int *)&a2[v77];
          if ( (_DWORD)v73 != -1 )
          {
            v74 = *(unsigned int *)(*(_QWORD *)(a1 + 16) + 4 * v73);
            v73 = (__int64)dest;
            *(_DWORD *)((char *)dest + v77) = v74;
          }
          v77 += 4;
        }
        while ( v77 != 4LL * (unsigned int)(a3 - 1) + 4 );
      }
      sub_2B310D0(a1 + 16, (__int64)&dest, v73, v74, v75, v76);
      if ( dest != v116 )
        _libc_free((unsigned __int64)dest);
    }
    else
    {
      sub_2B35330(a1 + 16, a2, &a2[4 * a3], a4, a5, a6);
    }
  }
  v78 = *(unsigned int *)(a1 + 24);
  if ( !(_DWORD)v78 )
    return *(_QWORD *)(a1 + 120);
  v80 = *(const void **)(a1 + 16);
  v81 = *(_QWORD *)(a1 + 80);
  if ( *(_DWORD *)(a1 + 88) == 2 )
    dest = *(void **)(v81 + 8);
  else
    dest = 0;
  v82 = sub_2B35AF0(a1, (__int64 *)v81, (__int64)&dest, v80, v78, a6);
  result = *(_QWORD *)(a1 + 120) + v82;
  if ( __OFADD__(*(_QWORD *)(a1 + 120), v82) )
  {
    result = 0x7FFFFFFFFFFFFFFFLL;
    if ( v82 <= 0 )
      return 0x8000000000000000LL;
  }
  return result;
}
