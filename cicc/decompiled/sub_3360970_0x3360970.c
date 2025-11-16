// Function: sub_3360970
// Address: 0x3360970
//
__int64 __fastcall sub_3360970(_QWORD *a1, __int64 a2, unsigned __int8 a3, unsigned __int8 a4, __int64 a5)
{
  __int64 v7; // rbx
  __int64 v8; // r15
  __int64 v9; // r14
  _QWORD *v10; // rax
  unsigned __int64 v11; // r14
  __int64 v12; // rdi
  __int64 v13; // rbx
  __int64 v14; // rdx
  __int64 v15; // rax
  unsigned __int64 v16; // rbx
  __int64 j; // rbx
  __int64 v18; // r14
  bool v19; // zf
  __int64 v20; // r9
  __int64 v21; // rax
  __int64 v22; // rdx
  __int64 v23; // rcx
  __int64 v24; // rdx
  unsigned int v25; // edi
  __int64 *v26; // rsi
  __int64 v27; // r9
  __m128i v28; // xmm1
  __int64 v29; // rsi
  __int64 v30; // r15
  unsigned int v31; // r8d
  unsigned int v32; // edi
  __int64 *v33; // rsi
  __int64 v34; // r9
  int v35; // r8d
  unsigned int v36; // edi
  __int64 *v37; // rsi
  __int64 v38; // r9
  __int64 v39; // rdx
  unsigned int v40; // edi
  __int64 *v41; // rsi
  __int64 v42; // r9
  __int64 v43; // r9
  __int64 v44; // rdx
  unsigned int v45; // esi
  __int64 *v46; // rdi
  __int64 v47; // r9
  __int64 v48; // r15
  __int64 v49; // r12
  __int64 v50; // rbx
  __int64 *v52; // r11
  int v53; // eax
  int v54; // eax
  __int64 i; // r14
  unsigned int v56; // ecx
  __int64 v57; // r15
  __int64 v58; // rsi
  unsigned int v59; // edx
  __int64 v60; // rax
  __int64 v61; // r8
  __int64 v62; // rdx
  __int64 v63; // rcx
  unsigned int v64; // esi
  __int64 v65; // r8
  __int64 v66; // r11
  unsigned int v67; // ecx
  __int64 *v68; // rax
  __int64 v69; // rdx
  int v70; // esi
  int v71; // r10d
  int v72; // edi
  int v73; // r10d
  __int64 v74; // rdi
  int v75; // esi
  int v76; // r10d
  int v77; // esi
  int v78; // r10d
  __int64 v79; // r9
  __int64 *v80; // rdi
  int v81; // ecx
  int v82; // edx
  int v83; // eax
  __int64 v84; // rax
  int v85; // r8d
  int v86; // r8d
  int v87; // edx
  __int64 *v88; // rdi
  __int64 v89; // r9
  unsigned int v90; // r11d
  __int64 v91; // rcx
  int v92; // r8d
  int v93; // r8d
  __int64 v94; // r9
  unsigned int v95; // edx
  __int64 v96; // r11
  int v97; // edi
  __int64 *v98; // rcx
  int v99; // r8d
  __int64 v100; // r10
  __int64 *v101; // rsi
  int v102; // edi
  unsigned int v103; // ecx
  int v104; // r8d
  __int64 v105; // r10
  unsigned int v106; // ecx
  int v107; // edi
  int v108; // [rsp+Ch] [rbp-84h]
  unsigned int v109; // [rsp+14h] [rbp-7Ch]
  unsigned int v110; // [rsp+14h] [rbp-7Ch]
  __int64 v111; // [rsp+18h] [rbp-78h]
  char *v112; // [rsp+40h] [rbp-50h] BYREF
  __m128i v113; // [rsp+48h] [rbp-48h] BYREF

  v7 = *a1;
  v8 = a1[1];
  v9 = *(_QWORD *)(*a1 + 584LL);
  v10 = *(_QWORD **)(v8 + 48);
  if ( v10 == *(_QWORD **)(v9 + 56) )
  {
    v11 = v9 + 48;
  }
  else
  {
    v11 = *v10 & 0xFFFFFFFFFFFFFFF8LL;
    if ( !v11 )
      BUG();
    if ( (*(_QWORD *)v11 & 4) == 0 && (*(_BYTE *)(v11 + 44) & 4) != 0 )
    {
      for ( i = *(_QWORD *)v11; ; i = *(_QWORD *)v11 )
      {
        v11 = i & 0xFFFFFFFFFFFFFFF8LL;
        if ( (*(_BYTE *)(v11 + 44) & 4) == 0 )
          break;
      }
    }
  }
  v12 = a1[1];
  if ( *(int *)(a2 + 24) < 0 )
    sub_37584B0(v12, a2, a3, a4, a5);
  else
    sub_3755B20(v12, a2, a3, a4, a5);
  v13 = *(_QWORD *)(v7 + 584);
  v14 = a1[1];
  v15 = *(_QWORD *)(v13 + 56);
  v16 = v13 + 48;
  if ( *(_QWORD *)(v14 + 48) != v15 )
  {
    v16 = **(_QWORD **)(v8 + 48) & 0xFFFFFFFFFFFFFFF8LL;
    if ( !v16 )
      BUG();
    if ( (*(_QWORD *)v16 & 4) == 0 && (*(_BYTE *)(v16 + 44) & 4) != 0 )
    {
      for ( j = *(_QWORD *)v16; ; j = *(_QWORD *)v16 )
      {
        v16 = j & 0xFFFFFFFFFFFFFFF8LL;
        if ( (*(_BYTE *)(v16 + 44) & 4) == 0 )
          break;
      }
    }
  }
  if ( v16 == v11 )
    return 0;
  if ( v11 == *(_QWORD *)(*a1 + 584LL) + 48LL )
  {
    v18 = *(_QWORD *)(*(_QWORD *)(v14 + 40) + 56LL);
  }
  else
  {
    if ( (*(_BYTE *)v11 & 4) == 0 && (*(_BYTE *)(v11 + 44) & 8) != 0 )
    {
      do
        v11 = *(_QWORD *)(v11 + 8);
      while ( (*(_BYTE *)(v11 + 44) & 8) != 0 );
    }
    v18 = *(_QWORD *)(v11 + 8);
  }
  v19 = sub_2E88ED0(v18, 0) == 0;
  v21 = *a1;
  v22 = *(_QWORD *)(*a1 + 592LL);
  if ( v19 )
    goto LABEL_68;
  if ( (*(_BYTE *)(*(_QWORD *)v22 + 904LL) & 1) != 0 )
  {
    v56 = *(_DWORD *)(v22 + 752);
    v57 = *(_QWORD *)(v21 + 32);
    if ( v56 )
    {
      v58 = *(_QWORD *)(v22 + 736);
      v59 = (v56 - 1) & (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4));
      v60 = v58 + 80LL * v59;
      v61 = *(_QWORD *)v60;
      if ( a2 == *(_QWORD *)v60 )
      {
LABEL_78:
        v62 = v58 + 80LL * v56;
        if ( v60 != v62 )
        {
          v112 = &v113.m128i_i8[8];
          v113.m128i_i64[0] = 0x100000000LL;
          v63 = *(unsigned int *)(v60 + 16);
          if ( (_DWORD)v63 )
            sub_335BF90((__int64)&v112, (char **)(v60 + 8), v62, v63, v61, v20);
LABEL_81:
          v64 = *(_DWORD *)(v57 + 712);
          v65 = v57 + 688;
          if ( v64 )
          {
            v66 = *(_QWORD *)(v57 + 696);
            v110 = ((unsigned int)v18 >> 9) ^ ((unsigned int)v18 >> 4);
            v67 = (v64 - 1) & v110;
            v68 = (__int64 *)(v66 + 32LL * v67);
            v69 = *v68;
            if ( v18 == *v68 )
            {
LABEL_83:
              if ( v112 != (char *)&v113.m128i_u64[1] )
                _libc_free((unsigned __int64)v112);
              v21 = *a1;
              v22 = *(_QWORD *)(*a1 + 592LL);
              goto LABEL_22;
            }
            v79 = 1;
            v80 = 0;
            while ( v69 != -4096 )
            {
              if ( v69 == -8192 && !v80 )
                v80 = v68;
              v67 = (v64 - 1) & (v79 + v67);
              v68 = (__int64 *)(v66 + 32LL * v67);
              v69 = *v68;
              if ( v18 == *v68 )
                goto LABEL_83;
              v79 = (unsigned int)(v79 + 1);
            }
            v81 = *(_DWORD *)(v57 + 704);
            if ( v80 )
              v68 = v80;
            ++*(_QWORD *)(v57 + 688);
            v82 = v81 + 1;
            if ( 4 * (v81 + 1) < 3 * v64 )
            {
              if ( v64 - *(_DWORD *)(v57 + 708) - v82 > v64 >> 3 )
              {
LABEL_110:
                *(_DWORD *)(v57 + 704) = v82;
                if ( *v68 != -4096 )
                  --*(_DWORD *)(v57 + 708);
                *v68 = v18;
                v68[1] = (__int64)(v68 + 3);
                v68[2] = 0x100000000LL;
                if ( v113.m128i_i32[0] )
                  sub_335BF90((__int64)(v68 + 1), &v112, v113.m128i_u32[0], 0x100000000LL, v65, v79);
                goto LABEL_83;
              }
              sub_2E7DD40(v57 + 688, v64);
              v99 = *(_DWORD *)(v57 + 712);
              if ( v99 )
              {
                v65 = (unsigned int)(v99 - 1);
                v100 = *(_QWORD *)(v57 + 696);
                v101 = 0;
                v102 = 1;
                v103 = v65 & v110;
                v82 = *(_DWORD *)(v57 + 704) + 1;
                v68 = (__int64 *)(v100 + 32LL * ((unsigned int)v65 & v110));
                v79 = *v68;
                if ( v18 == *v68 )
                  goto LABEL_110;
                while ( v79 != -4096 )
                {
                  if ( !v101 && v79 == -8192 )
                    v101 = v68;
                  v103 = v65 & (v102 + v103);
                  v68 = (__int64 *)(v100 + 32LL * v103);
                  v79 = *v68;
                  if ( v18 == *v68 )
                    goto LABEL_110;
                  ++v102;
                }
LABEL_135:
                if ( v101 )
                  v68 = v101;
                goto LABEL_110;
              }
              goto LABEL_177;
            }
          }
          else
          {
            ++*(_QWORD *)(v57 + 688);
          }
          sub_2E7DD40(v57 + 688, 2 * v64);
          v104 = *(_DWORD *)(v57 + 712);
          if ( v104 )
          {
            v65 = (unsigned int)(v104 - 1);
            v105 = *(_QWORD *)(v57 + 696);
            v106 = v65 & (((unsigned int)v18 >> 9) ^ ((unsigned int)v18 >> 4));
            v82 = *(_DWORD *)(v57 + 704) + 1;
            v68 = (__int64 *)(v105 + 32LL * v106);
            v79 = *v68;
            if ( v18 == *v68 )
              goto LABEL_110;
            v107 = 1;
            v101 = 0;
            while ( v79 != -4096 )
            {
              if ( !v101 && v79 == -8192 )
                v101 = v68;
              v106 = v65 & (v107 + v106);
              v68 = (__int64 *)(v105 + 32LL * v106);
              v79 = *v68;
              if ( v18 == *v68 )
                goto LABEL_110;
              ++v107;
            }
            goto LABEL_135;
          }
LABEL_177:
          ++*(_DWORD *)(v57 + 704);
          BUG();
        }
      }
      else
      {
        v83 = 1;
        while ( v61 != -4096 )
        {
          v20 = (unsigned int)(v83 + 1);
          v84 = (v56 - 1) & (v59 + v83);
          v59 = v84;
          v60 = v58 + 80 * v84;
          v61 = *(_QWORD *)v60;
          if ( a2 == *(_QWORD *)v60 )
            goto LABEL_78;
          v83 = v20;
        }
      }
    }
    v113 = 0;
    v112 = &v113.m128i_i8[8];
    v113.m128i_i32[1] = 1;
    goto LABEL_81;
  }
LABEL_22:
  v23 = *(_QWORD *)(v22 + 736);
  v24 = *(unsigned int *)(v22 + 752);
  if ( !(_DWORD)v24 )
    return v18;
  v25 = (v24 - 1) & (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4));
  v26 = (__int64 *)(v23 + 80LL * v25);
  v27 = *v26;
  if ( a2 == *v26 )
  {
LABEL_24:
    if ( v26 != (__int64 *)(v23 + 80LL * (unsigned int)v24) )
    {
      v28 = _mm_loadu_si128((const __m128i *)(v26 + 7));
      v29 = v26[7];
      if ( v29 )
      {
        v30 = *(_QWORD *)(v21 + 32);
        v113.m128i_i64[0] = v29;
        v113.m128i_i32[2] = v28.m128i_i32[2];
        v31 = *(_DWORD *)(v30 + 744);
        if ( v31 )
        {
          v111 = *(_QWORD *)(v30 + 728);
          v109 = ((unsigned int)v18 >> 9) ^ ((unsigned int)v18 >> 4);
          v32 = (v31 - 1) & v109;
          v33 = (__int64 *)(v111 + 24LL * v32);
          v34 = *v33;
          if ( v18 == *v33 )
            goto LABEL_28;
          v108 = 1;
          v52 = 0;
          while ( v34 != -4096 )
          {
            if ( !v52 && v34 == -8192 )
              v52 = v33;
            v32 = (v31 - 1) & (v108 + v32);
            v33 = (__int64 *)(v111 + 24LL * v32);
            v34 = *v33;
            if ( v18 == *v33 )
              goto LABEL_28;
            ++v108;
          }
          v53 = *(_DWORD *)(v30 + 736);
          if ( v52 )
            v33 = v52;
          ++*(_QWORD *)(v30 + 720);
          v54 = v53 + 1;
          if ( 4 * v54 < 3 * v31 )
          {
            if ( v31 - *(_DWORD *)(v30 + 740) - v54 > v31 >> 3 )
            {
LABEL_65:
              *(_DWORD *)(v30 + 736) = v54;
              if ( *v33 != -4096 )
                --*(_DWORD *)(v30 + 740);
              *v33 = v18;
              *(__m128i *)(v33 + 1) = _mm_loadu_si128(&v113);
              v21 = *a1;
              v22 = *(_QWORD *)(*a1 + 592LL);
LABEL_68:
              v23 = *(_QWORD *)(v22 + 736);
              v24 = *(unsigned int *)(v22 + 752);
              goto LABEL_28;
            }
            sub_2E7DF70(v30 + 720, v31);
            v85 = *(_DWORD *)(v30 + 744);
            if ( v85 )
            {
              v86 = v85 - 1;
              v87 = 1;
              v88 = 0;
              v89 = *(_QWORD *)(v30 + 728);
              v90 = v86 & v109;
              v33 = (__int64 *)(v89 + 24LL * (v86 & v109));
              v91 = *v33;
              v54 = *(_DWORD *)(v30 + 736) + 1;
              if ( v18 != *v33 )
              {
                while ( v91 != -4096 )
                {
                  if ( !v88 && v91 == -8192 )
                    v88 = v33;
                  v90 = v86 & (v87 + v90);
                  v33 = (__int64 *)(v89 + 24LL * v90);
                  v91 = *v33;
                  if ( v18 == *v33 )
                    goto LABEL_65;
                  ++v87;
                }
                if ( v88 )
                  v33 = v88;
              }
              goto LABEL_65;
            }
LABEL_178:
            ++*(_DWORD *)(v30 + 736);
            BUG();
          }
        }
        else
        {
          ++*(_QWORD *)(v30 + 720);
        }
        sub_2E7DF70(v30 + 720, 2 * v31);
        v92 = *(_DWORD *)(v30 + 744);
        if ( v92 )
        {
          v93 = v92 - 1;
          v94 = *(_QWORD *)(v30 + 728);
          v95 = v93 & (((unsigned int)v18 >> 9) ^ ((unsigned int)v18 >> 4));
          v33 = (__int64 *)(v94 + 24LL * v95);
          v96 = *v33;
          v54 = *(_DWORD *)(v30 + 736) + 1;
          if ( v18 != *v33 )
          {
            v97 = 1;
            v98 = 0;
            while ( v96 != -4096 )
            {
              if ( !v98 && v96 == -8192 )
                v98 = v33;
              v95 = v93 & (v97 + v95);
              v33 = (__int64 *)(v94 + 24LL * v95);
              v96 = *v33;
              if ( v18 == *v33 )
                goto LABEL_65;
              ++v97;
            }
            if ( v98 )
              v33 = v98;
          }
          goto LABEL_65;
        }
        goto LABEL_178;
      }
    }
  }
  else
  {
    v77 = 1;
    while ( v27 != -4096 )
    {
      v78 = v77 + 1;
      v25 = (v24 - 1) & (v25 + v77);
      v26 = (__int64 *)(v23 + 80LL * v25);
      v27 = *v26;
      if ( a2 == *v26 )
        goto LABEL_24;
      v77 = v78;
    }
  }
LABEL_28:
  if ( !(_DWORD)v24 )
    return v18;
  v35 = v24 - 1;
  v36 = (v24 - 1) & (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4));
  v37 = (__int64 *)(v23 + 80LL * v36);
  v38 = *v37;
  if ( a2 == *v37 )
  {
LABEL_30:
    if ( v37 != (__int64 *)(v23 + 80LL * (unsigned int)v24) )
    {
      if ( !*((_BYTE *)v37 + 72) )
        goto LABEL_35;
      *(_DWORD *)(v18 + 44) |= 0x8000u;
      v21 = *a1;
      v39 = *(_QWORD *)(*a1 + 592LL);
      v23 = *(_QWORD *)(v39 + 736);
      v24 = *(unsigned int *)(v39 + 752);
    }
  }
  else
  {
    v70 = 1;
    while ( v38 != -4096 )
    {
      v71 = v70 + 1;
      v36 = v35 & (v36 + v70);
      v37 = (__int64 *)(v23 + 80LL * v36);
      v38 = *v37;
      if ( a2 == *v37 )
        goto LABEL_30;
      v70 = v71;
    }
  }
  if ( !(_DWORD)v24 )
    return v18;
  v35 = v24 - 1;
LABEL_35:
  v40 = v35 & (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4));
  v41 = (__int64 *)(v23 + 80LL * v40);
  v42 = *v41;
  if ( a2 != *v41 )
  {
    v75 = 1;
    while ( v42 != -4096 )
    {
      v76 = v75 + 1;
      v40 = v35 & (v40 + v75);
      v41 = (__int64 *)(v23 + 80LL * v40);
      v42 = *v41;
      if ( a2 == *v41 )
        goto LABEL_36;
      v75 = v76;
    }
    goto LABEL_39;
  }
LABEL_36:
  if ( v41 == (__int64 *)(v23 + 80LL * (unsigned int)v24) )
  {
LABEL_39:
    if ( !(_DWORD)v24 )
      return v18;
    v35 = v24 - 1;
    goto LABEL_41;
  }
  v43 = v41[5];
  if ( v43 )
  {
    sub_2E882B0(v18, *(_QWORD *)(v21 + 32), v43);
    v21 = *a1;
    v44 = *(_QWORD *)(*a1 + 592LL);
    v23 = *(_QWORD *)(v44 + 736);
    v24 = *(unsigned int *)(v44 + 752);
    goto LABEL_39;
  }
LABEL_41:
  v45 = v35 & (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4));
  v46 = (__int64 *)(v23 + 80LL * v45);
  v47 = *v46;
  if ( a2 == *v46 )
  {
LABEL_42:
    if ( v46 != (__int64 *)(80 * v24 + v23) )
    {
      v48 = v46[6];
      if ( v48 )
      {
        v49 = v18;
        if ( (*(_BYTE *)v16 & 4) == 0 )
        {
          while ( (*(_BYTE *)(v16 + 44) & 8) != 0 )
            v16 = *(_QWORD *)(v16 + 8);
        }
        v50 = *(_QWORD *)(v16 + 8);
        if ( v18 != v50 )
        {
          while ( 1 )
          {
            sub_2E88680(v49, *(_QWORD *)(v21 + 32), v48);
            if ( !v49 )
              BUG();
            if ( (*(_BYTE *)v49 & 4) != 0 )
            {
              v49 = *(_QWORD *)(v49 + 8);
              if ( v49 == v50 )
                return v18;
            }
            else
            {
              while ( (*(_BYTE *)(v49 + 44) & 8) != 0 )
                v49 = *(_QWORD *)(v49 + 8);
              v49 = *(_QWORD *)(v49 + 8);
              if ( v49 == v50 )
                return v18;
            }
            v21 = *a1;
          }
        }
      }
    }
  }
  else
  {
    v72 = 1;
    while ( v47 != -4096 )
    {
      v73 = v72 + 1;
      v74 = v35 & (v45 + v72);
      v45 = v74;
      v46 = (__int64 *)(v23 + 80 * v74);
      v47 = *v46;
      if ( a2 == *v46 )
        goto LABEL_42;
      v72 = v73;
    }
  }
  return v18;
}
