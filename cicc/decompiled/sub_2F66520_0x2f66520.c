// Function: sub_2F66520
// Address: 0x2f66520
//
__int64 __fastcall sub_2F66520(__int64 a1, unsigned int a2, __int64 a3)
{
  unsigned __int64 v3; // r13
  __int64 v4; // rax
  __int64 v6; // rax
  __int64 *v7; // rbx
  unsigned __int64 v8; // r12
  __int64 *v9; // rdx
  __int64 v10; // rsi
  unsigned int v11; // edi
  unsigned int v12; // ecx
  __int64 v13; // rbx
  unsigned int *v14; // r9
  char v15; // r15
  unsigned int *v16; // rax
  unsigned __int64 v17; // rcx
  __int64 v18; // r12
  __int64 v19; // r9
  __int64 v20; // rdx
  __int64 *v21; // rax
  __int64 v22; // rdx
  __int64 v23; // rax
  unsigned int v24; // r12d
  bool v25; // al
  __int64 v26; // rdx
  __int64 v27; // r8
  __int64 v28; // r9
  __int64 v29; // rcx
  __int64 *v30; // rax
  __int128 v31; // rax
  __int64 v32; // rcx
  __int64 v33; // r8
  __int64 v34; // r9
  _QWORD *v35; // r12
  __int128 v36; // rax
  __int64 v37; // rsi
  __int64 *v38; // rax
  __int64 v39; // rdx
  __int64 v40; // rax
  unsigned __int64 v41; // rax
  unsigned int *v43; // r12
  unsigned int v44; // edx
  unsigned int v45; // eax
  _BYTE *v46; // rdx
  _BYTE *v47; // r12
  _BYTE *v48; // r15
  _BYTE *v49; // rbx
  bool v50; // al
  _BYTE *v51; // r15
  unsigned int v52; // esi
  _QWORD *v53; // r15
  __int64 v54; // rdx
  char v55; // al
  _QWORD *v56; // rdx
  bool v57; // al
  _DWORD *v58; // rax
  __int64 v59; // rax
  __int64 v60; // rdx
  __int64 v61; // r8
  __int64 v62; // rcx
  int v63; // r12d
  __int64 v64; // rsi
  __int64 v65; // rax
  int v66; // edx
  unsigned int *v67; // rax
  unsigned __int64 v68; // rax
  __int64 v69; // rdx
  __int64 v70; // rax
  char v71; // al
  char v72; // al
  __int64 v73; // rdx
  __int64 v74; // rdx
  __int64 v75; // rsi
  unsigned __int64 v76; // rcx
  __int64 v77; // rax
  __int64 v78; // rax
  __int64 v79; // rdx
  __int64 *v80; // rax
  __int128 v81; // rax
  __int64 v82; // rax
  _QWORD *v83; // r9
  __int64 *v84; // rdi
  bool v85; // [rsp+Fh] [rbp-81h]
  __int64 v86; // [rsp+10h] [rbp-80h]
  __int64 v87; // [rsp+18h] [rbp-78h]
  unsigned __int64 v88; // [rsp+18h] [rbp-78h]
  __int64 v89; // [rsp+20h] [rbp-70h]
  __int64 v90; // [rsp+20h] [rbp-70h]
  __int64 v91; // [rsp+20h] [rbp-70h]
  char v92; // [rsp+28h] [rbp-68h]
  unsigned __int64 v93; // [rsp+28h] [rbp-68h]
  __int64 v94; // [rsp+28h] [rbp-68h]
  __int64 v95; // [rsp+28h] [rbp-68h]
  __int64 v96; // [rsp+28h] [rbp-68h]
  _QWORD *v97; // [rsp+28h] [rbp-68h]
  __int64 v98; // [rsp+30h] [rbp-60h]
  _QWORD v100[2]; // [rsp+40h] [rbp-50h] BYREF
  __int64 v101; // [rsp+50h] [rbp-40h]

  v3 = *(_QWORD *)(a1 + 128) + ((unsigned __int64)a2 << 6);
  v98 = *(_QWORD *)(*(_QWORD *)(*(_QWORD *)a1 + 64LL) + 8LL * a2);
  v4 = *(_QWORD *)(v98 + 8);
  if ( (v4 & 0xFFFFFFFFFFFFFFF8LL) == 0 )
  {
    *(_QWORD *)(v3 + 8) = -1;
    v24 = 0;
    *(_QWORD *)(v3 + 16) = -1;
    return v24;
  }
  v92 = *(_BYTE *)(a1 + 32);
  if ( (v4 & 6) != 0 )
  {
    v6 = *(_QWORD *)((v4 & 0xFFFFFFFFFFFFFFF8LL) + 16);
    v89 = v6;
    if ( v92 )
    {
      *(_QWORD *)(v3 + 24) = 1;
      *(_QWORD *)(v3 + 32) = 0;
      *(__m128i *)(v3 + 8) = _mm_loadu_si128((const __m128i *)(v3 + 24));
      if ( *(_WORD *)(v6 + 68) == 10 )
      {
        *(_QWORD *)(v3 + 24) = 0;
        *(_BYTE *)(v3 + 56) = 1;
      }
    }
    else
    {
      v46 = *(_BYTE **)(v6 + 32);
      v47 = &v46[40 * (*(_DWORD *)(v6 + 40) & 0xFFFFFF)];
      if ( v46 == v47 )
        goto LABEL_104;
      v48 = *(_BYTE **)(v6 + 32);
      while ( 1 )
      {
        v49 = v48;
        v50 = sub_2DADC00(v48);
        if ( v50 )
          break;
        v48 += 40;
        if ( v47 == v48 )
          goto LABEL_104;
      }
      v85 = v50;
      if ( v48 == v47 )
      {
LABEL_104:
        *(_QWORD *)(v3 + 8) = 0;
        *(_QWORD *)(v3 + 16) = 0;
        *(__m128i *)(v3 + 24) = _mm_loadu_si128((const __m128i *)(v3 + 8));
      }
      else
      {
        v87 = 0;
        v86 = 0;
        do
        {
          if ( *((_DWORD *)v49 + 2) == *(_DWORD *)(a1 + 8) )
          {
            v52 = *(_DWORD *)(a1 + 12);
            v53 = *(_QWORD **)(a1 + 72);
            v54 = (*(_DWORD *)v49 >> 8) & 0xFFF;
            if ( v52 )
            {
              if ( (_DWORD)v54 )
                v54 = (*(unsigned int (__fastcall **)(_QWORD))(*v53 + 296LL))(*(_QWORD *)(a1 + 72));
              else
                v54 = v52;
            }
            v55 = v49[4];
            v56 = (_QWORD *)(v53[34] + 16 * v54);
            v86 |= *v56;
            v87 |= v56[1];
            if ( (v55 & 1) == 0 && (v55 & 2) == 0 )
            {
              if ( (v49[3] & 0x10) != 0 )
              {
                v57 = v92;
                if ( (*(_DWORD *)v49 & 0xFFF00) != 0 )
                  v57 = v85;
                v92 = v57;
              }
              else
              {
                v92 = v85;
              }
            }
          }
          v51 = v49 + 40;
          if ( v49 + 40 == v47 )
            break;
          while ( 1 )
          {
            v49 = v51;
            if ( sub_2DADC00(v51) )
              break;
            v51 += 40;
            if ( v47 == v51 )
              goto LABEL_71;
          }
        }
        while ( v47 != v51 );
LABEL_71:
        *(_QWORD *)(v3 + 8) = v86;
        *(_QWORD *)(v3 + 16) = v87;
        *(__m128i *)(v3 + 24) = _mm_loadu_si128((const __m128i *)(v3 + 8));
        if ( v92 )
        {
          sub_2F65960((__int64)v100, *(_QWORD *)a1, *(_QWORD *)(v98 + 8));
          v67 = (unsigned int *)v100[0];
          *(_QWORD *)(v3 + 40) = v100[0];
          if ( v67 )
          {
            sub_2F66F20(a1, *v67, a3);
            v68 = *(_QWORD *)(a1 + 128) + ((unsigned __int64)**(unsigned int **)(v3 + 40) << 6);
            v69 = *(_QWORD *)(v68 + 32);
            v70 = *(_QWORD *)(v68 + 24);
            *(_QWORD *)(v3 + 32) |= v69;
            *(_QWORD *)(v3 + 24) |= v70;
          }
        }
      }
      if ( *(_WORD *)(v89 + 68) == 10 )
        *(_BYTE *)(v3 + 56) = 1;
    }
  }
  else
  {
    if ( *(_BYTE *)(a1 + 32) )
    {
      v40 = 0;
      v39 = 1;
    }
    else
    {
      v38 = (__int64 *)(*(_QWORD *)(*(_QWORD *)(a1 + 72) + 272LL) + 16LL * *(unsigned int *)(a1 + 12));
      v39 = *v38;
      v40 = v38[1];
    }
    *(_QWORD *)(v3 + 8) = v39;
    *(_QWORD *)(v3 + 16) = v40;
    v89 = 0;
    *(__m128i *)(v3 + 24) = _mm_loadu_si128((const __m128i *)(v3 + 8));
  }
  v7 = *(__int64 **)a3;
  v8 = *(_QWORD *)(v98 + 8) & 0xFFFFFFFFFFFFFFF8LL;
  v9 = (__int64 *)sub_2E09D00(*(__int64 **)a3, v8);
  v10 = *v7 + 24LL * *((unsigned int *)v7 + 2);
  if ( v9 == (__int64 *)v10 )
  {
    *(_QWORD *)(v3 + 48) = 0;
    return 0;
  }
  v11 = *(_DWORD *)(v8 + 24);
  v12 = *(_DWORD *)((*v9 & 0xFFFFFFFFFFFFFFF8LL) + 24);
  if ( (unsigned __int64)(v12 | (*v9 >> 1) & 3) > v11 )
  {
    v15 = 0;
    v13 = 0;
    v14 = 0;
    if ( v11 < v12 )
      goto LABEL_12;
  }
  else
  {
    v13 = v9[1];
    v14 = (unsigned int *)v9[2];
    v15 = 0;
    if ( v8 == (v13 & 0xFFFFFFFFFFFFFFF8LL) )
    {
      if ( (__int64 *)v10 == v9 + 3 )
      {
        v15 = 1;
        goto LABEL_12;
      }
      v15 = 1;
      v12 = *(_DWORD *)((v9[3] & 0xFFFFFFFFFFFFFFF8LL) + 24);
      v9 += 3;
    }
    if ( v8 == *((_QWORD *)v14 + 1) )
      v14 = 0;
    if ( v11 < v12 )
      goto LABEL_12;
  }
  v43 = (unsigned int *)v9[2];
  v13 = v9[1];
  if ( v43 != v14 && v43 )
  {
    v44 = *(_DWORD *)((*((_QWORD *)v43 + 1) & 0xFFFFFFFFFFFFFFF8LL) + 24) | (*((__int64 *)v43 + 1) >> 1) & 3;
    v45 = *(_DWORD *)((*(_QWORD *)(v98 + 8) & 0xFFFFFFFFFFFFFFF8LL) + 24) | (*(__int64 *)(v98 + 8) >> 1) & 3;
    if ( v44 < v45 )
    {
      sub_2F66F20(a3, *v43, a1);
    }
    else if ( v44 > v45 && v14 )
    {
      *(_QWORD *)(v3 + 48) = v14;
      return 5;
    }
    *(_QWORD *)(v3 + 48) = v43;
    v41 = *(_QWORD *)(a3 + 128) + ((unsigned __int64)*v43 << 6);
    if ( (*(_QWORD *)(v41 + 8) || *(_QWORD *)(v41 + 16)) && *(_DWORD *)(*(_QWORD *)(a3 + 80) + 4LL * *v43) != -1 )
    {
      v24 = 2;
      if ( (*(_BYTE *)(v98 + 8) & 6) != 0 )
        return (*(_OWORD *)(v3 + 24) & *(_OWORD *)(v41 + 24)) == 0 ? 2 : 5;
      return v24;
    }
    return 0;
  }
LABEL_12:
  *(_QWORD *)(v3 + 48) = v14;
  if ( !v14 )
    return 0;
  sub_2F66F20(a3, *v14, a1);
  v16 = *(unsigned int **)(v3 + 48);
  v17 = *(_QWORD *)(a3 + 128) + ((unsigned __int64)*v16 << 6);
  if ( *(_BYTE *)(v17 + 56) )
  {
    v18 = *(_QWORD *)((*((_QWORD *)v16 + 1) & 0xFFFFFFFFFFFFFFF8LL) + 16);
    v19 = *(_QWORD *)(v18 + 24);
    if ( v89 )
    {
      if ( v19 != *(_QWORD *)(v89 + 24) )
        goto LABEL_16;
      v88 = *(_QWORD *)(a3 + 128) + ((unsigned __int64)*v16 << 6);
      v96 = *(_QWORD *)(v18 + 24);
      v71 = sub_2F65890(
              *(_QWORD *)a1,
              *(_QWORD *)(*(_QWORD *)(*(_QWORD *)(*(_QWORD *)(a1 + 56) + 32LL) + 152LL)
                        + 16LL * *(unsigned int *)(v19 + 24)));
      v19 = v96;
      v17 = v88;
      if ( v71 )
        goto LABEL_16;
    }
    v97 = (_QWORD *)v17;
    v72 = sub_2E31A70(v19);
    v17 = (unsigned __int64)v97;
    if ( v72 )
    {
LABEL_16:
      v20 = *(_QWORD *)(a1 + 72);
      *(_BYTE *)(v17 + 56) = 0;
      v21 = (__int64 *)(*(_QWORD *)(v20 + 272) + 16LL * ((**(_DWORD **)(v18 + 32) >> 8) & 0xFFF));
      v22 = *v21;
      v23 = v21[1];
      *(_QWORD *)(v17 + 24) = v22;
      *(_QWORD *)(v17 + 32) = v23;
    }
    else
    {
      v73 = ~v97[1];
      v97[4] &= ~v97[2];
      v97[3] &= v73;
    }
  }
  if ( (*(_BYTE *)(v98 + 8) & 6) == 0 )
    return 3;
  v93 = v17;
  v24 = 1;
  if ( *(_WORD *)(v89 + 68) == 10 )
    return v24;
  v25 = sub_2F66340(*(_DWORD **)(a1 + 48), v89);
  v29 = v93;
  if ( v25 )
  {
    v74 = *(_QWORD *)(v93 + 24) | ~*(_QWORD *)(v3 + 8);
    *(_QWORD *)(v3 + 32) &= *(_QWORD *)(v93 + 32) | ~*(_QWORD *)(v3 + 16);
    *(_QWORD *)(v3 + 24) &= v74;
    return v24;
  }
  if ( v15 )
  {
    v26 = *(_DWORD *)((v13 & 0xFFFFFFFFFFFFFFF8LL) + 24) | (unsigned int)(v13 >> 1) & 3;
    if ( (unsigned int)v26 <= (*(_DWORD *)((*(_QWORD *)(v98 + 8) & 0xFFFFFFFFFFFFFFF8LL) + 24)
                             | (unsigned int)(*(__int64 *)(v98 + 8) >> 1) & 3) )
      return 0;
  }
  if ( *(_WORD *)(v89 + 68) == 20 )
  {
    v58 = *(_DWORD **)(v89 + 32);
    if ( (*v58 & 0xFFF00) == 0 && (v58[10] & 0xFFF00) == 0 && !*(_BYTE *)(*(_QWORD *)(a1 + 48) + 24LL) )
    {
      v90 = v93;
      v94 = *(_QWORD *)(v3 + 48);
      v59 = sub_2F64650(a1, v98, v26, v29, v27, v28);
      v62 = v90;
      v63 = v60;
      if ( v94 == v59 && *(_DWORD *)(a3 + 8) == (_DWORD)v60 )
        goto LABEL_91;
      v64 = v94;
      v91 = v59;
      v95 = v62;
      v65 = sub_2F64650(a3, v64, v60, v62, v61, v59);
      v29 = v95;
      if ( v91 )
      {
        if ( !v65 || *(_QWORD *)(v65 + 8) != *(_QWORD *)(v91 + 8) )
          goto LABEL_23;
      }
      else if ( v65 )
      {
        goto LABEL_23;
      }
      if ( v66 == v63 )
      {
LABEL_91:
        *(_BYTE *)(v3 + 59) = 1;
        return 1;
      }
    }
  }
LABEL_23:
  if ( *(_BYTE *)(a1 + 32) )
    return 3;
  v24 = 3;
  if ( (*(_OWORD *)(v3 + 8) & *(_OWORD *)(v29 + 24)) != 0 )
  {
    if ( v15 )
      return 5;
    v24 = 5;
    v30 = (__int64 *)(*(_QWORD *)(*(_QWORD *)(a1 + 72) + 272LL) + 16LL * *(unsigned int *)(a3 + 12));
    *(_QWORD *)&v31 = sub_2F612A0(*v30, v30[1], ~*(_QWORD *)(v3 + 8));
    if ( v31 != 0 )
    {
      if ( *(_BYTE *)(a1 + 33) )
      {
        v35 = *(_QWORD **)(sub_2DF8570(*(_QWORD *)(a1 + 56), *(_DWORD *)(a3 + 8), *((__int64 *)&v31 + 1), v32, v33, v34)
                         + 104);
        if ( !v35 )
        {
          v80 = (__int64 *)(*(_QWORD *)(*(_QWORD *)(a1 + 72) + 272LL) + 16LL * *(unsigned int *)(a3 + 12));
          *(_QWORD *)&v81 = sub_2F612A0(*v80, v80[1], *(_QWORD *)(v3 + 8));
          return v81 == 0 ? 3 : 5;
        }
        while ( 1 )
        {
          v37 = *(unsigned int *)(a3 + 12);
          if ( (_DWORD)v37 )
            *(_QWORD *)&v36 = (*(__int64 (__fastcall **)(_QWORD, __int64, _QWORD, _QWORD))(**(_QWORD **)(a1 + 72) + 312LL))(
                                *(_QWORD *)(a1 + 72),
                                v37,
                                v35[14],
                                v35[15]);
          else
            v36 = *((_OWORD *)v35 + 7);
          if ( (*(_OWORD *)(v3 + 8) & v36) != 0 )
          {
            sub_2F65960((__int64)v100, (__int64)v35, *(_QWORD *)(v98 + 8));
            if ( v100[0] )
            {
              if ( (*(_DWORD *)((v101 & 0xFFFFFFFFFFFFFFF8LL) + 24) | (unsigned int)(v101 >> 1) & 3) > (*(_DWORD *)((*(_QWORD *)(v98 + 8) & 0xFFFFFFFFFFFFFFF8LL) + 24) | (unsigned int)(*(__int64 *)(v98 + 8) >> 1) & 3) )
                break;
            }
          }
          v35 = (_QWORD *)v35[13];
          if ( !v35 )
            return 3;
        }
      }
      else
      {
        v75 = *(_QWORD *)(a1 + 64);
        v76 = *(_QWORD *)(v98 + 8) & 0xFFFFFFFFFFFFFFF8LL;
        v77 = *(_QWORD *)(v76 + 16);
        if ( v77 )
        {
          v78 = *(_QWORD *)(v77 + 24);
        }
        else
        {
          v82 = *(unsigned int *)(v75 + 304);
          v83 = *(_QWORD **)(v75 + 296);
          if ( *(_DWORD *)(v75 + 304) )
          {
            do
            {
              v84 = &v83[2 * (v82 >> 1)];
              if ( (*(_DWORD *)(v76 + 24) | (unsigned int)(*(__int64 *)(v98 + 8) >> 1) & 3) >= (*(_DWORD *)((*v84 & 0xFFFFFFFFFFFFFFF8LL) + 24)
                                                                                              | (unsigned int)(*v84 >> 1)
                                                                                              & 3) )
              {
                v83 = v84 + 2;
                v82 = v82 - (v82 >> 1) - 1;
              }
              else
              {
                v82 >>= 1;
              }
            }
            while ( v82 > 0 );
          }
          v78 = *(v83 - 1);
        }
        v79 = *(_QWORD *)(*(_QWORD *)(v75 + 152) + 16LL * *(unsigned int *)(v78 + 24) + 8);
        if ( (*(_DWORD *)((v13 & 0xFFFFFFFFFFFFFFF8LL) + 24) | (unsigned int)(v13 >> 1) & 3) < (*(_DWORD *)((v79 & 0xFFFFFFFFFFFFFFF8LL) + 24)
                                                                                              | (unsigned int)(v79 >> 1)
                                                                                              & 3) )
          return 4;
      }
      return 5;
    }
  }
  return v24;
}
