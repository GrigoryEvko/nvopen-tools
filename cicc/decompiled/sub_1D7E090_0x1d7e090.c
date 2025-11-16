// Function: sub_1D7E090
// Address: 0x1d7e090
//
__int64 __fastcall sub_1D7E090(__int64 a1, __int64 a2)
{
  __int64 (*v3)(void); // rdx
  __int64 v4; // rax
  __int64 (*v5)(void); // rdx
  __int64 v6; // rax
  __int64 v7; // r15
  unsigned __int64 v8; // rsi
  unsigned __int64 v9; // rax
  unsigned int v10; // r12d
  __int64 v11; // r14
  __int64 v12; // r12
  __int64 i; // r15
  unsigned __int16 *v14; // r13
  unsigned __int16 *j; // rdx
  unsigned __int64 v16; // rcx
  _QWORD *v17; // rax
  unsigned __int64 v18; // rdx
  unsigned __int64 v19; // r15
  __int64 k; // r15
  unsigned __int64 v21; // rdx
  unsigned __int64 v22; // r12
  __int64 m; // r12
  __int16 *v24; // rdx
  __int16 v25; // ax
  int v26; // r13d
  __int64 v27; // rsi
  __int64 n; // rax
  __int64 v29; // rdx
  __int64 v30; // rcx
  int v31; // ecx
  __int64 v32; // rdx
  __int64 v33; // rdx
  unsigned int v34; // edi
  unsigned __int64 v35; // rcx
  __int64 v36; // rax
  __int64 v37; // rdx
  unsigned int v38; // r10d
  __int64 v39; // rcx
  __int64 v40; // r8
  __int64 v41; // r9
  unsigned __int64 *v42; // rsi
  unsigned __int64 v43; // r14
  int v44; // eax
  __int64 v45; // rdx
  int v46; // ecx
  __int64 v47; // r8
  __int64 v48; // r9
  unsigned __int64 v49; // rax
  __int64 v50; // rax
  __int64 v51; // rdi
  __int64 v52; // r10
  __int64 v53; // rax
  int v54; // edx
  _QWORD *v55; // r9
  __int64 v56; // r13
  unsigned int v57; // eax
  __int16 v58; // r8
  _WORD *v59; // rcx
  unsigned int v60; // edx
  unsigned __int16 *v61; // rax
  unsigned __int16 v62; // r8
  unsigned __int16 *v63; // r11
  unsigned __int16 *v64; // rsi
  unsigned __int16 *v65; // rax
  unsigned __int64 v66; // rcx
  __int64 v67; // rax
  _QWORD *v68; // r11
  int v69; // r11d
  unsigned __int16 *v70; // rax
  __int64 v71; // rdx
  __int64 v72; // rsi
  __int64 v73; // rdx
  _QWORD *v74; // rax
  int v75; // eax
  __int64 v76; // rdi
  __int64 (*v77)(); // rax
  __int16 v79; // ax
  __int64 v80; // rax
  void *v81; // r12
  int v82; // edx
  int v83; // edx
  __int64 v84; // rax
  unsigned __int16 v85; // cx
  unsigned int v86; // eax
  __int64 v87; // r14
  __int64 v88; // r13
  const __m128i *v89; // rax
  __int64 v90; // rsi
  __int64 v91; // rax
  __int64 v92; // rax
  __int64 v93; // [rsp+8h] [rbp-98h]
  _QWORD *v94; // [rsp+10h] [rbp-90h]
  unsigned __int16 v95; // [rsp+18h] [rbp-88h]
  unsigned __int8 v96; // [rsp+27h] [rbp-79h]
  _QWORD *v97; // [rsp+28h] [rbp-78h]
  __int64 v98; // [rsp+38h] [rbp-68h]
  _QWORD *v99; // [rsp+38h] [rbp-68h]
  _OWORD v100[2]; // [rsp+40h] [rbp-60h] BYREF
  __int64 v101; // [rsp+60h] [rbp-40h]

  v96 = sub_1636880(a1, *(_QWORD *)a2);
  if ( v96 )
    return 0;
  *(_QWORD *)(a1 + 240) = *(_QWORD *)(a2 + 40);
  v3 = *(__int64 (**)(void))(**(_QWORD **)(a2 + 16) + 112LL);
  v4 = 0;
  if ( v3 != sub_1D00B10 )
    v4 = v3();
  *(_QWORD *)(a1 + 232) = v4;
  v5 = *(__int64 (**)(void))(**(_QWORD **)(a2 + 16) + 40LL);
  v6 = 0;
  if ( v5 != sub_1D00B00 )
    v6 = v5();
  *(_QWORD *)(a1 + 248) = v6;
  v94 = (_QWORD *)(a2 + 320);
  *(_BYTE *)(a1 + 280) = (unsigned int)(*(_DWORD *)(*(_QWORD *)(a2 + 16) + 40LL) - 34) <= 1;
  v97 = (_QWORD *)(*(_QWORD *)(a2 + 320) & 0xFFFFFFFFFFFFFFF8LL);
  if ( (_QWORD *)(a2 + 320) == v97 )
    goto LABEL_98;
  v93 = a1 + 256;
  do
  {
    v7 = *(_QWORD *)(a1 + 240);
    if ( v93 != v7 + 304 )
    {
      v8 = *(unsigned int *)(v7 + 320);
      v9 = *(_QWORD *)(a1 + 264);
      *(_DWORD *)(a1 + 272) = v8;
      v10 = (unsigned int)(v8 + 63) >> 6;
      v11 = v10;
      if ( v8 > v9 << 6 )
      {
        v81 = (void *)malloc(8LL * v10);
        if ( !v81 )
        {
          if ( 8 * v11 || (v92 = malloc(1u)) == 0 )
            sub_16BD1C0("Allocation failed", 1u);
          else
            v81 = (void *)v92;
        }
        memcpy(v81, *(const void **)(v7 + 304), 8 * v11);
        _libc_free(*(_QWORD *)(a1 + 256));
        *(_QWORD *)(a1 + 256) = v81;
        *(_QWORD *)(a1 + 264) = v11;
        goto LABEL_12;
      }
      if ( (_DWORD)v8 )
      {
        memcpy(*(void **)(a1 + 256), *(const void **)(v7 + 304), 8LL * v10);
        v82 = *(_DWORD *)(a1 + 272);
        v9 = *(_QWORD *)(a1 + 264);
        v10 = (unsigned int)(v82 + 63) >> 6;
        v11 = v10;
        if ( v9 > v10 )
        {
LABEL_113:
          v84 = v9 - v11;
          if ( v84 )
            memset((void *)(*(_QWORD *)(a1 + 256) + 8 * v11), 0, 8 * v84);
          v82 = *(_DWORD *)(a1 + 272);
        }
        v83 = v82 & 0x3F;
        if ( v83 )
          *(_QWORD *)(*(_QWORD *)(a1 + 256) + 8LL * (v10 - 1)) &= ~(-1LL << v83);
        goto LABEL_12;
      }
      if ( v9 > v10 )
        goto LABEL_113;
    }
LABEL_12:
    v12 = v97[12];
    for ( i = v97[11]; v12 != i; i += 8 )
    {
      v14 = *(unsigned __int16 **)(*(_QWORD *)i + 160LL);
      for ( j = (unsigned __int16 *)sub_1DD77D0(); v14 != j; *v17 |= 1LL << v16 )
      {
        v16 = *j;
        j += 4;
        v17 = (_QWORD *)(*(_QWORD *)(a1 + 256) + ((v16 >> 3) & 0x1FF8));
      }
    }
    v98 = v97[3];
    v18 = v98 & 0xFFFFFFFFFFFFFFF8LL;
    if ( (v98 & 0xFFFFFFFFFFFFFFF8LL) == 0 )
      BUG();
    v19 = v98 & 0xFFFFFFFFFFFFFFF8LL;
    if ( (*(_QWORD *)v18 & 4) == 0 && (*(_BYTE *)(v18 + 46) & 4) != 0 )
    {
      for ( k = *(_QWORD *)v18; ; k = *(_QWORD *)v19 )
      {
        v19 = k & 0xFFFFFFFFFFFFFFF8LL;
        if ( (*(_BYTE *)(v19 + 46) & 4) == 0 )
          break;
      }
    }
    v99 = v97 + 3;
    if ( v97 + 3 == (_QWORD *)v19 )
      goto LABEL_97;
    while ( 2 )
    {
      v21 = *(_QWORD *)v19 & 0xFFFFFFFFFFFFFFF8LL;
      if ( !v21 )
        BUG();
      v22 = *(_QWORD *)v19 & 0xFFFFFFFFFFFFFFF8LL;
      if ( (*(_QWORD *)v21 & 4) == 0 && (*(_BYTE *)(v21 + 46) & 4) != 0 )
      {
        for ( m = *(_QWORD *)v21; ; m = *(_QWORD *)v22 )
        {
          v22 = m & 0xFFFFFFFFFFFFFFF8LL;
          if ( (*(_BYTE *)(v22 + 46) & 4) == 0 )
            break;
        }
      }
      v24 = *(__int16 **)(v19 + 16);
      v25 = *v24;
      if ( *v24 != 1 )
      {
        if ( *(_BYTE *)(a1 + 280) )
        {
          v79 = *(_WORD *)(v19 + 46);
          if ( (v79 & 4) != 0 || (v79 & 8) == 0 )
            v80 = (*((_QWORD *)v24 + 1) >> 17) & 1LL;
          else
            LOBYTE(v80) = sub_1E15D00(v19, 0x20000, 1);
          if ( (_BYTE)v80 && !(unsigned __int8)sub_1E178F0(v19) )
          {
            v76 = *(_QWORD *)(a1 + 248);
            v77 = *(__int64 (**)())(*(_QWORD *)v76 + 200LL);
            if ( v77 == sub_1D7DF80 )
              goto LABEL_96;
            v86 = ((__int64 (__fastcall *)(__int64, unsigned __int64))v77)(v76, v19);
            if ( !v86 )
              goto LABEL_96;
            v87 = 0;
            v88 = 40LL * v86;
            while ( 1 )
            {
              v89 = (const __m128i *)(v87 + *(_QWORD *)(v19 + 32));
              v100[0] = _mm_loadu_si128(v89);
              v100[1] = _mm_loadu_si128(v89 + 1);
              v101 = v89[2].m128i_i64[0];
              if ( v89->m128i_i8[0] )
                break;
              if ( (BYTE4(v100[0]) & 1) == 0 )
              {
                v90 = v89->m128i_u32[2];
                if ( (int)v90 >= 0 )
                  break;
                v91 = sub_1E69D00(*(_QWORD *)(a1 + 240), v90);
                if ( !v91 || **(_WORD **)(v91 + 16) != 9 )
                  break;
              }
              v87 += 40;
              if ( v88 == v87 )
                goto LABEL_96;
            }
          }
          v25 = **(_WORD **)(v19 + 16);
        }
        if ( v25 != 24 )
        {
          LOBYTE(v100[0]) = 0;
          if ( (unsigned __int8)sub_1E17B50(v19, 0, v100) || **(_WORD **)(v19 + 16) == 45 || !**(_WORD **)(v19 + 16) )
          {
            v26 = *(_DWORD *)(v19 + 40);
            if ( v26 )
            {
              v27 = *(_QWORD *)(v19 + 32);
              for ( n = v27; v27 + 40LL * (unsigned int)(v26 - 1) + 40 != n; n += 40 )
              {
                if ( !*(_BYTE *)n && (*(_BYTE *)(n + 3) & 0x10) != 0 )
                {
                  v31 = *(_DWORD *)(n + 8);
                  if ( v31 > 0 )
                  {
                    v29 = 1LL << v31;
                    v30 = (unsigned int)v31 >> 6;
                    if ( (*(_QWORD *)(*(_QWORD *)(a1 + 256) + 8 * v30) & v29) != 0
                      || (*(_QWORD *)(*(_QWORD *)(*(_QWORD *)(a1 + 240) + 304LL) + 8 * v30) & v29) != 0 )
                    {
                      goto LABEL_49;
                    }
                  }
                  else
                  {
                    v32 = *(_QWORD *)(a1 + 240);
                    if ( v31 )
                      v33 = *(_QWORD *)(*(_QWORD *)(v32 + 24) + 16LL * (v31 & 0x7FFFFFFF) + 8);
                    else
                      v33 = **(_QWORD **)(v32 + 272);
                    while ( v33 )
                    {
                      if ( (*(_BYTE *)(v33 + 3) & 0x10) == 0 && (*(_BYTE *)(v33 + 4) & 8) == 0 )
                        goto LABEL_49;
                      v33 = *(_QWORD *)(v33 + 32);
                    }
                  }
                }
              }
            }
LABEL_96:
            sub_1E162E0(v19);
            v96 = 1;
            if ( v99 != (_QWORD *)v22 )
              goto LABEL_72;
            break;
          }
        }
      }
      v26 = *(_DWORD *)(v19 + 40);
      if ( !v26 )
        goto LABEL_71;
      v27 = *(_QWORD *)(v19 + 32);
LABEL_49:
      v34 = 0;
      while ( 1 )
      {
        v36 = v27 + 40LL * v34;
        if ( *(_BYTE *)v36 )
          break;
        if ( (*(_BYTE *)(v36 + 3) & 0x10) != 0 )
        {
          v35 = *(unsigned int *)(v36 + 8);
          if ( (int)v35 > 0 )
          {
            v71 = *(_QWORD *)(a1 + 232);
            if ( !v71 )
              BUG();
            v72 = *(_QWORD *)(v71 + 56) + 2LL * *(unsigned int *)(*(_QWORD *)(v71 + 8) + 24LL * (unsigned int)v35 + 4);
            while ( 1 )
            {
              v73 = v72;
              if ( !v72 )
                break;
              while ( 1 )
              {
                v73 += 2;
                v74 = (_QWORD *)(*(_QWORD *)(a1 + 256) + ((v35 >> 3) & 0x1FF8));
                *v74 &= ~(1LL << v35);
                v75 = *(unsigned __int16 *)(v73 - 2);
                v72 = 0;
                if ( !(_WORD)v75 )
                  break;
                v35 = (unsigned int)(v75 + v35);
                if ( !v73 )
                  goto LABEL_52;
              }
            }
          }
        }
LABEL_52:
        if ( ++v34 == v26 )
          goto LABEL_65;
LABEL_53:
        v27 = *(_QWORD *)(v19 + 32);
      }
      if ( *(_BYTE *)v36 != 12 )
        goto LABEL_52;
      v37 = *(_QWORD *)(v36 + 24);
      v38 = (unsigned int)(*(_DWORD *)(a1 + 272) + 31) >> 5;
      if ( (unsigned int)(*(_DWORD *)(a1 + 272) + 31) <= 0x3F )
      {
        LODWORD(v40) = 0;
      }
      else
      {
        v39 = 0;
        v40 = ((v38 - 2) >> 1) + 1;
        v41 = 8 * v40;
        do
        {
          v42 = (unsigned __int64 *)(v39 + *(_QWORD *)(a1 + 256));
          v43 = *v42 & ~(unsigned __int64)(unsigned int)~*(_DWORD *)(v37 + v39);
          v44 = *(_DWORD *)(v37 + v39 + 4);
          v39 += 8;
          *v42 = v43 & ~((unsigned __int64)(unsigned int)~v44 << 32);
        }
        while ( v41 != v39 );
        v37 += v41;
        v38 &= 1u;
      }
      if ( !v38 )
        goto LABEL_52;
      v45 = v37 + 4;
      v46 = 0;
      v47 = 8LL * (unsigned int)v40;
      v48 = v45;
      while ( 1 )
      {
        v49 = (unsigned __int64)(unsigned int)~*(_DWORD *)(v45 - 4) << v46;
        v46 += 32;
        *(_QWORD *)(v47 + *(_QWORD *)(a1 + 256)) &= ~v49;
        if ( v45 == v48 )
          break;
        v45 += 4;
      }
      if ( ++v34 != v26 )
        goto LABEL_53;
LABEL_65:
      v50 = *(unsigned int *)(v19 + 40);
      if ( (_DWORD)v50 )
      {
        v51 = 0;
        v52 = 40 * v50;
        do
        {
          v53 = v51 + *(_QWORD *)(v19 + 32);
          if ( !*(_BYTE *)v53 && (*(_BYTE *)(v53 + 3) & 0x10) == 0 )
          {
            v54 = *(_DWORD *)(v53 + 8);
            if ( v54 > 0 )
            {
              v55 = *(_QWORD **)(a1 + 232);
              if ( !v55 )
                BUG();
              v56 = v55[7];
              v57 = *(_DWORD *)(v55[1] + 24LL * (unsigned int)v54 + 16);
              v58 = v54 * (v57 & 0xF);
              v59 = (_WORD *)(v56 + 2LL * (v57 >> 4));
              v95 = 0;
              v60 = 0;
              v61 = v59 + 1;
              v62 = *v59 + v58;
LABEL_77:
              v63 = v61;
              v64 = v61;
              if ( v61 )
              {
                while ( 1 )
                {
                  v65 = (unsigned __int16 *)(v55[6] + 4LL * v62);
                  v66 = *v65;
                  v60 = v65[1];
                  if ( (_WORD)v66 )
                    break;
LABEL_118:
                  v85 = *v63;
                  v61 = 0;
                  ++v63;
                  if ( !v85 )
                    goto LABEL_77;
                  v62 += v85;
                  v64 = v63;
                  if ( !v63 )
                    goto LABEL_120;
                }
                while ( 1 )
                {
                  v67 = v56 + 2LL * *(unsigned int *)(v55[1] + 24LL * (unsigned __int16)v66 + 8);
                  if ( v67 )
                    break;
                  if ( !(_WORD)v60 )
                  {
                    v95 = v66;
                    goto LABEL_118;
                  }
                  v66 = v60;
                  v60 = 0;
                }
              }
              else
              {
LABEL_120:
                v66 = v95;
                v67 = 0;
              }
              while ( v64 )
              {
                while ( 1 )
                {
                  v67 += 2;
                  v68 = (_QWORD *)(*(_QWORD *)(a1 + 256) + ((v66 >> 3) & 0x1FF8));
                  *v68 |= 1LL << v66;
                  v69 = *(unsigned __int16 *)(v67 - 2);
                  if ( !(_WORD)v69 )
                    break;
                  v66 = (unsigned int)(v69 + v66);
                }
                if ( (_WORD)v60 )
                {
                  v67 = v55[7] + 2LL * *(unsigned int *)(v55[1] + 24LL * (unsigned __int16)v60 + 8);
                  v66 = v60;
                  v60 = 0;
                }
                else
                {
                  v60 = *v64;
                  v62 += v60;
                  if ( (_WORD)v60 )
                  {
                    ++v64;
                    v70 = (unsigned __int16 *)(v55[6] + 4LL * v62);
                    v66 = *v70;
                    v60 = v70[1];
                    v67 = v55[7] + 2LL * *(unsigned int *)(v55[1] + 24LL * (unsigned __int16)v66 + 8);
                  }
                  else
                  {
                    v67 = 0;
                    v64 = 0;
                  }
                }
              }
            }
          }
          v51 += 40;
        }
        while ( v52 != v51 );
      }
LABEL_71:
      if ( v99 != (_QWORD *)v22 )
      {
LABEL_72:
        v19 = v22;
        continue;
      }
      break;
    }
LABEL_97:
    v97 = (_QWORD *)(*v97 & 0xFFFFFFFFFFFFFFF8LL);
  }
  while ( v94 != v97 );
LABEL_98:
  *(_DWORD *)(a1 + 272) = 0;
  return v96;
}
