// Function: sub_1687790
// Address: 0x1687790
//
size_t __fastcall sub_1687790(size_t a1, unsigned __int64 a2, __int64 a3, __int64 a4)
{
  size_t v4; // r15
  size_t v5; // r14
  __int16 v6; // ax
  __int64 v7; // r8
  unsigned int v8; // eax
  unsigned int v9; // r10d
  _QWORD *v10; // rbx
  int v11; // r9d
  size_t v12; // r12
  unsigned int v13; // ecx
  int v14; // r13d
  unsigned int v15; // ecx
  unsigned int v16; // ebx
  __int64 v17; // rax
  __int64 v18; // rsi
  _QWORD *v19; // rdi
  _QWORD *v20; // rax
  int v21; // edx
  __int64 v22; // r8
  int v23; // r9d
  unsigned int v24; // ecx
  __int64 v25; // rdx
  __int64 v26; // rax
  unsigned __int64 v27; // rax
  unsigned __int64 v28; // r12
  size_t result; // rax
  __int64 v30; // rax
  __int64 v31; // rdx
  __int64 v32; // rax
  __int64 v33; // rax
  unsigned __int64 *v34; // r12
  __int64 v35; // rax
  __int64 v36; // r13
  __int64 v37; // rax
  __int64 v38; // rbx
  size_t *v39; // r12
  __int64 v40; // rax
  unsigned __int64 *v41; // r12
  __int64 v42; // rdi
  __int64 v43; // r12
  int v44; // r14d
  __int64 v45; // rdi
  int v46; // edx
  int v47; // ecx
  int v48; // r8d
  int v49; // r9d
  _QWORD *v50; // r13
  int v51; // eax
  __int64 v52; // r12
  unsigned __int64 v53; // rbx
  __int64 v54; // rax
  unsigned int v55; // ebx
  unsigned int v56; // eax
  __int64 v57; // rdx
  _DWORD **v58; // r14
  unsigned int v59; // ecx
  int v60; // r12d
  __int64 v61; // r13
  __int64 v62; // rdi
  __int64 v63; // rax
  _QWORD *v64; // rax
  int v65; // edx
  __int64 v66; // r8
  int v67; // r9d
  unsigned int v68; // ecx
  const void *v69; // r11
  void *v70; // rdi
  __int64 v71; // rax
  bool v72; // zf
  __int64 v73; // r12
  __int64 v74; // rax
  _QWORD *v75; // rax
  int v76; // edx
  int v77; // ecx
  int v78; // r9d
  __int64 v79; // r8
  unsigned int v80; // r10d
  void *v81; // rdi
  _QWORD *v82; // rbx
  unsigned int v83; // r10d
  __int64 v84; // r8
  __int64 v85; // rdi
  int v86; // esi
  _QWORD *v87; // rax
  int v88; // edx
  int v89; // ecx
  int v90; // r9d
  void *src; // [rsp+0h] [rbp-60h]
  __int64 v92; // [rsp+8h] [rbp-58h]
  int v93; // [rsp+8h] [rbp-58h]
  int v94; // [rsp+10h] [rbp-50h]
  int v95; // [rsp+10h] [rbp-50h]
  __int64 v96; // [rsp+10h] [rbp-50h]
  int v97; // [rsp+18h] [rbp-48h]
  int v98; // [rsp+18h] [rbp-48h]
  int v99; // [rsp+18h] [rbp-48h]
  __int64 i; // [rsp+18h] [rbp-48h]
  int v101; // [rsp+18h] [rbp-48h]
  unsigned int v102; // [rsp+18h] [rbp-48h]
  unsigned int v103; // [rsp+18h] [rbp-48h]
  __int64 v104; // [rsp+18h] [rbp-48h]
  __int64 v105; // [rsp+18h] [rbp-48h]
  __int64 v106; // [rsp+20h] [rbp-40h]
  __int64 v107; // [rsp+20h] [rbp-40h]
  _DWORD **v108; // [rsp+20h] [rbp-40h]
  __int64 v109; // [rsp+20h] [rbp-40h]
  size_t v110; // [rsp+20h] [rbp-40h]
  __int64 v111; // [rsp+20h] [rbp-40h]
  unsigned int v112; // [rsp+20h] [rbp-40h]
  int v113; // [rsp+20h] [rbp-40h]
  unsigned int v114; // [rsp+28h] [rbp-38h]
  __int64 v115; // [rsp+28h] [rbp-38h]
  unsigned int v116; // [rsp+28h] [rbp-38h]

  v4 = a1;
  v5 = a2;
  v6 = *(_WORD *)(a1 + 84) >> 4;
  if ( (_BYTE)v6 == 1 )
  {
    v31 = *(_QWORD *)(a1 + 104);
    a1 = (unsigned int)(a2 >> 11) ^ (unsigned int)(a2 >> 8) ^ (unsigned int)(a2 >> 5);
    v32 = *(_DWORD *)(v4 + 40) & ((unsigned int)(a2 >> 11) ^ (unsigned int)(a2 >> 8) ^ (unsigned int)(a2 >> 5));
    v114 = (a2 >> 11) ^ (a2 >> 8) ^ (a2 >> 5);
    a3 = *(_QWORD *)(v31 + 8 * v32);
    v7 = 8 * v32;
    if ( a3 )
    {
      while ( 1 )
      {
        v33 = *(unsigned int *)(a3 + 4);
        a3 += 4;
        if ( (_DWORD)v33 == -1 )
          break;
        v34 = (unsigned __int64 *)(*(_QWORD *)(v4 + 88) + 8 * v33);
        result = *v34;
        if ( a2 == *v34 )
        {
          *v34 = a2;
          return result;
        }
      }
    }
  }
  else if ( (_BYTE)v6 == 2 )
  {
    v114 = a2;
    v7 = 8LL * (*(_DWORD *)(a1 + 40) & (unsigned int)a2);
    a3 = *(_QWORD *)(*(_QWORD *)(a1 + 104) + v7);
    if ( a3 )
    {
      while ( 1 )
      {
        v40 = *(unsigned int *)(a3 + 4);
        a3 += 4;
        if ( (_DWORD)v40 == -1 )
          break;
        v41 = (unsigned __int64 *)(*(_QWORD *)(a1 + 88) + 8 * v40);
        result = *v41;
        if ( a2 == *v41 )
        {
          *v41 = a2;
          return result;
        }
      }
    }
  }
  else
  {
    v114 = 0;
    v7 = 0;
    if ( !(_BYTE)v6 )
    {
      a2 = *(_QWORD *)(a1 + 32);
      a1 = v5;
      v114 = a2
           ? (*(__int64 (__fastcall **)(size_t, unsigned __int64, __int64, __int64, _QWORD))(v4 + 16))(
               v5,
               a2,
               a3,
               a4,
               0)
           : (*(unsigned __int64 (__fastcall **)(size_t, _QWORD, __int64, __int64, _QWORD))v4)(v5, 0, a3, a4, 0);
      a3 = *(_QWORD *)(v4 + 104);
      v7 = 8LL * (*(_DWORD *)(v4 + 40) & v114);
      v35 = *(_QWORD *)(a3 + v7);
      if ( v35 )
      {
        v36 = v35 + 4;
        v37 = *(unsigned int *)(v35 + 4);
        v38 = *(_QWORD *)(v4 + 88);
        if ( (_DWORD)v37 != -1 )
        {
          do
          {
            v39 = (size_t *)(v38 + 8 * v37);
            a2 = v5;
            a1 = *v39;
            if ( *(_QWORD *)(v4 + 32) )
            {
              if ( (*(unsigned __int8 (__fastcall **)(size_t, size_t))(v4 + 24))(a1, v5) )
                goto LABEL_36;
            }
            else if ( (*(unsigned __int8 (__fastcall **)(size_t, size_t))(v4 + 8))(a1, v5) )
            {
LABEL_36:
              result = *v39;
              *v39 = v5;
              return result;
            }
            v37 = *(unsigned int *)(v36 + 4);
            v36 += 4;
          }
          while ( (_DWORD)v37 != -1 );
          v7 = 8LL * (*(_DWORD *)(v4 + 40) & v114);
        }
      }
    }
  }
  v8 = *(_DWORD *)(v4 + 72);
  v9 = *(_DWORD *)(v4 + 80);
  v10 = *(_QWORD **)(v4 + 96);
  if ( v8 >= v9 )
  {
LABEL_18:
    if ( v8 )
    {
      a3 = v8;
      v30 = 0;
      while ( 1 )
      {
        v11 = v30;
        v12 = 4 * v30;
        v13 = ~*((_DWORD *)v10 + v30);
        if ( *((_DWORD *)v10 + v30) != -1 )
          break;
        if ( a3 == ++v30 )
          goto LABEL_59;
      }
    }
    else
    {
LABEL_59:
      LODWORD(v71) = *(_DWORD *)(v4 + 80);
      do
        v71 = (unsigned int)(2 * v71);
      while ( v9 >= (unsigned int)v71 );
      v72 = (*(_BYTE *)(v4 + 84) & 0xC) == 0;
      *(_DWORD *)(v4 + 80) = v71;
      v73 = 4 * v71;
      v103 = v9;
      v111 = v7;
      if ( v72 )
      {
        v85 = (__int64)v10;
        v86 = 4 * v71;
        v87 = sub_1685950(v10, 4 * v71);
        v84 = v111;
        v83 = v103;
        v82 = v87;
        if ( !v87 )
        {
          sub_1683C30(v85, v86, v88, v89, v111, v90, (char)src);
          v83 = v103;
          v84 = v111;
        }
        *(_QWORD *)(v4 + 96) = v82;
        v12 = 4LL * v83;
      }
      else
      {
        v74 = sub_1689050(a1, a2, a3);
        v75 = sub_1685080(*(_QWORD *)(v74 + 24), v73);
        v79 = v111;
        v80 = v103;
        v81 = v75;
        if ( !v75 )
        {
          sub_1683C30(0, v73, v76, v77, v111, v78, (char)src);
          v81 = 0;
          v80 = v103;
          v79 = v111;
        }
        *(_QWORD *)(v4 + 96) = v81;
        v12 = 4LL * v80;
        v104 = v79;
        v112 = v80;
        memcpy(v81, v10, v12);
        *(_BYTE *)(v4 + 84) &= 0xF3u;
        v82 = *(_QWORD **)(v4 + 96);
        v83 = v112;
        v84 = v104;
      }
      a1 = (size_t)v82 + v12;
      v105 = v84;
      v113 = v83;
      memset((char *)v82 + v12, 0, 4LL * (*(_DWORD *)(v4 + 80) - v83));
      v7 = v105;
      v13 = -1;
      v11 = v113;
    }
  }
  else
  {
    v11 = *(_DWORD *)(v4 + 72);
    v12 = 4LL * v8;
    while ( 1 )
    {
      v13 = ~*(_DWORD *)((char *)v10 + v12);
      if ( *(_DWORD *)((char *)v10 + v12) != -1 )
        break;
      ++v11;
      v12 += 4LL;
      if ( v9 == v11 )
        goto LABEL_18;
    }
  }
  _BitScanForward((unsigned int *)&v14, v13);
  v15 = *(_DWORD *)(v4 + 76);
  v16 = 32 * v11 + v14;
  if ( v16 >= v15 )
  {
    LODWORD(v17) = *(_DWORD *)(v4 + 76);
    do
      v17 = (unsigned int)(2 * v17);
    while ( v16 >= (unsigned int)v17 );
    *(_DWORD *)(v4 + 76) = v17;
    v18 = 8 * v17;
    if ( (*(_BYTE *)(v4 + 84) & 3) != 0 )
    {
      src = *(void **)(v4 + 88);
      v95 = v15;
      v101 = v11;
      v109 = v7;
      v92 = 8 * v17;
      v63 = sub_1689050(a1, v18, a3);
      v64 = sub_1685080(*(_QWORD *)(v63 + 24), v92);
      v66 = v109;
      v67 = v101;
      v68 = v95;
      v69 = src;
      v70 = v64;
      if ( !v64 )
      {
        sub_1683C30(0, v92, v65, v95, v109, v101, (char)src);
        v69 = src;
        v70 = 0;
        v68 = v95;
        v67 = v101;
        v66 = v109;
      }
      *(_QWORD *)(v4 + 88) = v70;
      v93 = v67;
      v96 = v66;
      v102 = v68;
      v110 = 8LL * v68;
      memcpy(v70, v69, v110);
      *(_BYTE *)(v4 + 84) &= 0xFCu;
      v20 = *(_QWORD **)(v4 + 88);
      v25 = v110;
      v24 = v102;
      v22 = v96;
      v23 = v93;
    }
    else
    {
      v19 = *(_QWORD **)(v4 + 88);
      v94 = v15;
      v97 = v11;
      v106 = v7;
      v20 = sub_1685950(v19, v18);
      v22 = v106;
      v23 = v97;
      v24 = v94;
      if ( !v20 )
      {
        sub_1683C30((__int64)v19, v18, v21, v94, v106, v97, (char)src);
        v20 = 0;
        v24 = v94;
        v23 = v97;
        v22 = v106;
      }
      *(_QWORD *)(v4 + 88) = v20;
      v25 = 8LL * v24;
    }
    v98 = v23;
    v107 = v22;
    memset((char *)v20 + v25, 0, 8LL * (*(_DWORD *)(v4 + 76) - v24));
    v11 = v98;
    v7 = v107;
  }
  v99 = v11;
  v108 = (_DWORD **)(v7 + *(_QWORD *)(v4 + 104));
  *v108 = sub_1687300(*v108, v16, (__int64)v108);
  *(_DWORD *)(*(_QWORD *)(v4 + 96) + v12) |= 1 << v14;
  v26 = *(_QWORD *)(v4 + 88);
  *(_DWORD *)(v4 + 72) = v99;
  *(_QWORD *)(v26 + 8LL * v16) = v5;
  v27 = *(_QWORD *)(v4 + 48);
  v28 = *(_QWORD *)(v4 + 64);
  *(_DWORD *)(v4 + 56) ^= v114;
  *(_QWORD *)(v4 + 48) = ++v27;
  if ( v27 > v28 )
  {
    v42 = 2 * v28;
    v115 = 2 * v28;
    v43 = 8LL * (unsigned int)(2 * *(_DWORD *)(v4 + 40) + 2);
    v44 = 2 * *(_DWORD *)(v4 + 40) + 1;
    v45 = *(_QWORD *)(sub_1689050(v42, v16, v108) + 24);
    v50 = sub_1685080(v45, v43);
    if ( !v50 )
      sub_1683C30(v45, v43, v46, v47, v48, v49, (char)src);
    memset(v50, 0, v43);
    v51 = *(_DWORD *)(v4 + 40);
    if ( v51 >= 0 )
    {
      v52 = 8LL * v51;
      v53 = 8 * (v51 - (unsigned __int64)(unsigned int)v51);
      do
      {
        sub_16856A0(*(_QWORD **)(*(_QWORD *)(v4 + 104) + v52));
        *(_QWORD *)(*(_QWORD *)(v4 + 104) + v52) = 0;
        v54 = v52;
        v52 -= 8;
      }
      while ( v53 != v54 );
    }
    sub_16856A0(*(_QWORD **)(v4 + 104));
    *(_DWORD *)(v4 + 40) = v44;
    *(_QWORD *)(v4 + 104) = v50;
    *(_QWORD *)(v4 + 64) = v115;
    for ( i = 0; *(_DWORD *)(v4 + 80) > (unsigned int)i; ++i )
    {
      v55 = *(_DWORD *)(*(_QWORD *)(v4 + 96) + 4 * i);
      if ( v55 )
      {
        do
        {
          v116 = v55;
          _BitScanForward(&v59, v55);
          v60 = 1 << v59;
          v61 = v59 + 32 * (_DWORD)i;
          v55 ^= 1 << v59;
          v62 = *(_QWORD *)(*(_QWORD *)(v4 + 88) + 8 * v61);
          if ( *(_QWORD *)(v4 + 32) )
            v56 = (*(__int64 (__fastcall **)(__int64))(v4 + 16))(v62);
          else
            v56 = (*(__int64 (__fastcall **)(__int64))v4)(v62);
          v57 = *(_QWORD *)(v4 + 104);
          v58 = (_DWORD **)(v57 + 8LL * (*(_DWORD *)(v4 + 40) & v56));
          *v58 = sub_1687300(*v58, (unsigned int)v61, v57);
        }
        while ( v60 != v116 );
      }
    }
  }
  return 0;
}
