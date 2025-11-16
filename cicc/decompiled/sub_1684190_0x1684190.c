// Function: sub_1684190
// Address: 0x1684190
//
__int64 __fastcall sub_1684190(__int64 a1, unsigned __int64 a2, __int64 a3)
{
  __int16 v5; // ax
  __int64 v6; // rcx
  unsigned int v7; // eax
  unsigned int v8; // r11d
  _DWORD *v9; // r10
  int v10; // r15d
  size_t v11; // r12
  unsigned int v12; // ebx
  unsigned int v13; // r11d
  unsigned int v14; // eax
  unsigned int v15; // ebx
  unsigned int v16; // eax
  __int64 v17; // rsi
  void *v18; // rdi
  __int64 v19; // rax
  int v20; // edx
  int v21; // r8d
  int v22; // r9d
  __int64 v23; // rcx
  unsigned int v24; // r11d
  __int64 v25; // rdx
  unsigned __int64 *v26; // r10
  unsigned __int64 v27; // rax
  unsigned __int64 v28; // r12
  __int64 result; // rax
  __int64 v30; // rdx
  size_t v31; // rax
  __int64 v32; // rax
  __int64 v33; // r12
  __int64 *v34; // r12
  __int64 v35; // rsi
  __int64 v36; // rax
  __int64 v37; // r12
  __int64 v38; // rbx
  __int64 v39; // r15
  __int64 v40; // rdi
  __int64 v41; // rax
  __int64 v42; // r12
  _QWORD *v43; // r12
  __int64 v44; // r12
  size_t v45; // r15
  int v46; // esi
  __int64 v47; // rdi
  int v48; // edx
  int v49; // ecx
  int v50; // r8d
  int v51; // r9d
  void *v52; // r13
  int v53; // eax
  __int64 v54; // r15
  unsigned __int64 v55; // rbx
  __int64 v56; // rax
  unsigned int v57; // ebx
  unsigned int v58; // eax
  _DWORD **v59; // r15
  unsigned int v60; // ecx
  int v61; // r12d
  __int64 v62; // r13
  __int64 v63; // rdi
  __int64 v64; // rax
  __int64 v65; // rax
  int v66; // edx
  int v67; // r9d
  __int64 v68; // rcx
  unsigned int v69; // r11d
  const void *v70; // r8
  void *v71; // rdi
  __int64 v72; // rax
  __int64 v73; // r12
  __int64 v74; // rax
  __int64 v75; // rax
  int v76; // edx
  int v77; // r8d
  int v78; // r9d
  __int64 v79; // rcx
  unsigned int v80; // r11d
  const void *v81; // r10
  void *v82; // rdi
  __int64 v83; // rbx
  unsigned int v84; // r11d
  __int64 v85; // rcx
  void *v86; // rdi
  int v87; // esi
  __int64 v88; // rdi
  __int64 v89; // rax
  int v90; // edx
  int v91; // r8d
  int v92; // r9d
  char v93; // [rsp+0h] [rbp-70h]
  void *src; // [rsp+8h] [rbp-68h]
  __int64 v95; // [rsp+10h] [rbp-60h]
  __int64 v96; // [rsp+10h] [rbp-60h]
  unsigned int v97; // [rsp+18h] [rbp-58h]
  unsigned int v98; // [rsp+18h] [rbp-58h]
  unsigned int v99; // [rsp+18h] [rbp-58h]
  void *v100; // [rsp+18h] [rbp-58h]
  __int64 v101; // [rsp+20h] [rbp-50h]
  __int64 v102; // [rsp+20h] [rbp-50h]
  _DWORD **v103; // [rsp+20h] [rbp-50h]
  __int64 v104; // [rsp+20h] [rbp-50h]
  size_t v105; // [rsp+20h] [rbp-50h]
  unsigned int v106; // [rsp+20h] [rbp-50h]
  __int64 v107; // [rsp+20h] [rbp-50h]
  __int64 v108; // [rsp+20h] [rbp-50h]
  unsigned int v109; // [rsp+20h] [rbp-50h]
  char v110; // [rsp+28h] [rbp-48h]
  __int64 i; // [rsp+28h] [rbp-48h]
  __int64 v112; // [rsp+28h] [rbp-48h]
  unsigned int v113; // [rsp+28h] [rbp-48h]
  unsigned int v114; // [rsp+28h] [rbp-48h]
  __int64 v115; // [rsp+28h] [rbp-48h]
  unsigned int v117; // [rsp+3Ch] [rbp-34h]
  int v118; // [rsp+3Ch] [rbp-34h]
  unsigned int v119; // [rsp+3Ch] [rbp-34h]

  v5 = *(_WORD *)(a1 + 84) >> 4;
  if ( (_BYTE)v5 == 1 )
  {
    v117 = (a2 >> 11) ^ (a2 >> 8) ^ (a2 >> 5);
    v6 = 8LL * (*(_DWORD *)(a1 + 40) & v117);
    v32 = *(_QWORD *)(*(_QWORD *)(a1 + 104) + v6);
    if ( v32 )
    {
      while ( 1 )
      {
        v33 = *(unsigned int *)(v32 + 4);
        v32 += 4;
        if ( (_DWORD)v33 == -1 )
          break;
        v34 = (__int64 *)(*(_QWORD *)(a1 + 88) + 16 * v33);
        if ( a2 == *v34 )
        {
LABEL_26:
          result = v34[1];
          v34[1] = a3;
          return result;
        }
      }
    }
  }
  else if ( (_BYTE)v5 == 2 )
  {
    v117 = a2;
    v6 = 8LL * (*(_DWORD *)(a1 + 40) & (unsigned int)a2);
    v41 = *(_QWORD *)(*(_QWORD *)(a1 + 104) + v6);
    if ( v41 )
    {
      while ( 1 )
      {
        v42 = *(unsigned int *)(v41 + 4);
        v41 += 4;
        if ( (_DWORD)v42 == -1 )
          break;
        v43 = (_QWORD *)(*(_QWORD *)(a1 + 88) + 16 * v42);
        if ( a2 == *v43 )
        {
          result = v43[1];
          v43[1] = a3;
          return result;
        }
      }
    }
  }
  else
  {
    v117 = 0;
    v6 = 0;
    if ( !(_BYTE)v5 )
    {
      v35 = *(_QWORD *)(a1 + 32);
      v117 = v35
           ? (*(__int64 (__fastcall **)(unsigned __int64, __int64, __int64, _QWORD))(a1 + 16))(a2, v35, a3, 0)
           : (*(__int64 (__fastcall **)(unsigned __int64, _QWORD, __int64, _QWORD))a1)(a2, 0, a3, 0);
      v6 = 8LL * (*(_DWORD *)(a1 + 40) & v117);
      v36 = *(_QWORD *)(*(_QWORD *)(a1 + 104) + v6);
      if ( v36 )
      {
        v37 = *(unsigned int *)(v36 + 4);
        v38 = *(_QWORD *)(a1 + 88);
        v39 = v36 + 4;
        if ( (_DWORD)v37 != -1 )
        {
          do
          {
            while ( 1 )
            {
              v34 = (__int64 *)(v38 + 16 * v37);
              v40 = *v34;
              if ( !*(_QWORD *)(a1 + 32) )
                break;
              if ( (*(unsigned __int8 (__fastcall **)(__int64, unsigned __int64))(a1 + 24))(v40, a2) )
                goto LABEL_26;
              v37 = *(unsigned int *)(v39 + 4);
              v39 += 4;
              if ( (_DWORD)v37 == -1 )
                goto LABEL_37;
            }
            if ( (*(unsigned __int8 (__fastcall **)(__int64, unsigned __int64))(a1 + 8))(v40, a2) )
              goto LABEL_26;
            v37 = *(unsigned int *)(v39 + 4);
            v39 += 4;
          }
          while ( (_DWORD)v37 != -1 );
LABEL_37:
          v6 = 8LL * (*(_DWORD *)(a1 + 40) & v117);
        }
      }
    }
  }
  v7 = *(_DWORD *)(a1 + 72);
  v8 = *(_DWORD *)(a1 + 80);
  v9 = *(_DWORD **)(a1 + 96);
  if ( v7 >= v8 )
  {
LABEL_18:
    if ( v7 )
    {
      v30 = v7;
      v31 = 0;
      while ( 1 )
      {
        v10 = v31;
        v11 = v31;
        v12 = ~v9[v31];
        if ( v9[v31] != -1 )
          break;
        if ( v30 == ++v31 )
          goto LABEL_60;
      }
    }
    else
    {
LABEL_60:
      LODWORD(v72) = *(_DWORD *)(a1 + 80);
      do
        v72 = (unsigned int)(2 * v72);
      while ( v8 >= (unsigned int)v72 );
      *(_DWORD *)(a1 + 80) = v72;
      v73 = 4 * v72;
      if ( (*(_BYTE *)(a1 + 84) & 0xC) != 0 )
      {
        v100 = v9;
        v106 = v8;
        v112 = v6;
        v74 = sub_1689050();
        v75 = sub_1685080(*(_QWORD *)(v74 + 24), v73);
        v79 = v112;
        v80 = v106;
        v81 = v100;
        v82 = (void *)v75;
        if ( !v75 )
        {
          sub_1683C30(0, v73, v76, v112, v77, v78, v93);
          v81 = v100;
          v82 = 0;
          v80 = v106;
          v79 = v112;
        }
        *(_QWORD *)(a1 + 96) = v82;
        v11 = v80;
        v107 = v79;
        v113 = v80;
        memcpy(v82, v81, v11 * 4);
        *(_BYTE *)(a1 + 84) &= 0xF3u;
        v83 = *(_QWORD *)(a1 + 96);
        v84 = v113;
        v85 = v107;
      }
      else
      {
        v87 = 4 * v72;
        v88 = (__int64)v9;
        v109 = v8;
        v115 = v6;
        v89 = sub_1685950(v9, 4 * v72);
        v85 = v115;
        v84 = v109;
        v83 = v89;
        if ( !v89 )
        {
          sub_1683C30(v88, v87, v90, v115, v91, v92, v93);
          v84 = v109;
          v85 = v115;
        }
        *(_QWORD *)(a1 + 96) = v83;
        v11 = v84;
      }
      v86 = (void *)(v83 + v11 * 4);
      v108 = v85;
      v114 = v84;
      v12 = -1;
      memset(v86, 0, 4LL * (*(_DWORD *)(a1 + 80) - v84));
      v6 = v108;
      v10 = v114;
    }
  }
  else
  {
    v10 = *(_DWORD *)(a1 + 72);
    v11 = v7;
    while ( 1 )
    {
      v12 = ~v9[v11];
      if ( v9[v11] != -1 )
        break;
      ++v10;
      ++v11;
      if ( v8 == v10 )
        goto LABEL_18;
    }
  }
  v13 = *(_DWORD *)(a1 + 76);
  _BitScanForward(&v14, v12);
  v15 = 32 * v10 + v14;
  v110 = v14;
  if ( v15 >= v13 )
  {
    v16 = *(_DWORD *)(a1 + 76);
    do
      v16 *= 2;
    while ( v15 >= v16 );
    *(_DWORD *)(a1 + 76) = v16;
    v17 = 16LL * v16;
    if ( (*(_BYTE *)(a1 + 84) & 3) != 0 )
    {
      src = *(void **)(a1 + 88);
      v98 = v13;
      v104 = v6;
      v95 = 16LL * v16;
      v64 = sub_1689050();
      v65 = sub_1685080(*(_QWORD *)(v64 + 24), v95);
      v68 = v104;
      v69 = v98;
      v70 = src;
      v71 = (void *)v65;
      if ( !v65 )
      {
        sub_1683C30(0, v95, v66, v104, (int)src, v67, v93);
        v70 = src;
        v71 = 0;
        v69 = v98;
        v68 = v104;
      }
      *(_QWORD *)(a1 + 88) = v71;
      v96 = v68;
      v99 = v69;
      v105 = 16LL * v69;
      memcpy(v71, v70, v105);
      *(_BYTE *)(a1 + 84) &= 0xFCu;
      v19 = *(_QWORD *)(a1 + 88);
      v25 = v105;
      v24 = v99;
      v23 = v96;
    }
    else
    {
      v18 = *(void **)(a1 + 88);
      v97 = v13;
      v101 = v6;
      v19 = sub_1685950(v18, v17);
      v23 = v101;
      v24 = v97;
      if ( !v19 )
      {
        sub_1683C30((__int64)v18, v17, v20, v101, v21, v22, v93);
        v19 = 0;
        v24 = v97;
        v23 = v101;
      }
      *(_QWORD *)(a1 + 88) = v19;
      v25 = 16LL * v24;
    }
    v102 = v23;
    memset((void *)(v19 + v25), 0, 16LL * (*(_DWORD *)(a1 + 76) - v24));
    v6 = v102;
  }
  v103 = (_DWORD **)(*(_QWORD *)(a1 + 104) + v6);
  *v103 = sub_1683EF0(*v103, v15);
  *(_DWORD *)(*(_QWORD *)(a1 + 96) + v11 * 4) |= 1 << v110;
  v26 = (unsigned __int64 *)(*(_QWORD *)(a1 + 88) + 16LL * v15);
  *(_DWORD *)(a1 + 72) = v10;
  *v26 = a2;
  v26[1] = a3;
  v27 = *(_QWORD *)(a1 + 48);
  v28 = *(_QWORD *)(a1 + 64);
  *(_DWORD *)(a1 + 56) ^= v117;
  *(_QWORD *)(a1 + 48) = ++v27;
  if ( v27 > v28 )
  {
    v44 = 2 * v28;
    v45 = 8LL * (unsigned int)(2 * *(_DWORD *)(a1 + 40) + 2);
    v118 = 2 * *(_DWORD *)(a1 + 40) + 1;
    v46 = 8 * (2 * *(_DWORD *)(a1 + 40) + 2);
    v47 = *(_QWORD *)(sub_1689050() + 24);
    v52 = (void *)sub_1685080(v47, v45);
    if ( !v52 )
      sub_1683C30(v47, v46, v48, v49, v50, v51, v93);
    memset(v52, 0, v45);
    v53 = *(_DWORD *)(a1 + 40);
    if ( v53 >= 0 )
    {
      v54 = 8LL * v53;
      v55 = 8 * (v53 - (unsigned __int64)(unsigned int)v53);
      do
      {
        sub_16856A0(*(_QWORD *)(*(_QWORD *)(a1 + 104) + v54));
        *(_QWORD *)(*(_QWORD *)(a1 + 104) + v54) = 0;
        v56 = v54;
        v54 -= 8;
      }
      while ( v55 != v56 );
    }
    sub_16856A0(*(_QWORD *)(a1 + 104));
    *(_QWORD *)(a1 + 64) = v44;
    *(_QWORD *)(a1 + 104) = v52;
    *(_DWORD *)(a1 + 40) = v118;
    for ( i = 0; *(_DWORD *)(a1 + 80) > (unsigned int)i; ++i )
    {
      v57 = *(_DWORD *)(*(_QWORD *)(a1 + 96) + 4 * i);
      if ( v57 )
      {
        do
        {
          v119 = v57;
          _BitScanForward(&v60, v57);
          v61 = 1 << v60;
          v62 = v60 + 32 * (_DWORD)i;
          v57 ^= 1 << v60;
          v63 = *(_QWORD *)(*(_QWORD *)(a1 + 88) + 16 * v62);
          if ( *(_QWORD *)(a1 + 32) )
            v58 = (*(__int64 (__fastcall **)(__int64))(a1 + 16))(v63);
          else
            v58 = (*(__int64 (__fastcall **)(__int64))a1)(v63);
          v59 = (_DWORD **)(*(_QWORD *)(a1 + 104) + 8LL * (*(_DWORD *)(a1 + 40) & v58));
          *v59 = sub_1683EF0(*v59, v62);
        }
        while ( v61 != v119 );
      }
    }
  }
  return 0;
}
