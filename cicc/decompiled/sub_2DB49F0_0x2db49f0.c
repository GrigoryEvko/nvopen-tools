// Function: sub_2DB49F0
// Address: 0x2db49f0
//
__int64 __fastcall sub_2DB49F0(_QWORD *a1, __int64 a2, int a3)
{
  int v3; // r13d
  _QWORD *v5; // rax
  __int64 v6; // r12
  __int64 v7; // r15
  unsigned int v8; // r14d
  _QWORD *v10; // rax
  int v11; // eax
  __int64 v12; // rdi
  __int64 (*v13)(); // rax
  __int64 v14; // rcx
  __int64 v15; // r8
  __int64 v16; // r9
  __int64 v17; // r10
  __int64 v18; // rax
  unsigned __int64 v19; // r15
  int v20; // eax
  __int64 v21; // r13
  __int64 v22; // r12
  int v23; // r10d
  unsigned int v24; // eax
  __int64 v25; // rdx
  __int64 v26; // rdx
  __int64 v27; // rsi
  __int64 v28; // rax
  __int64 v29; // r8
  __int64 v30; // rdi
  __int64 v31; // rax
  unsigned __int64 v32; // r15
  _QWORD *v33; // rax
  _QWORD *v34; // rdx
  __int64 v35; // rax
  __int64 i; // r15
  _QWORD *v37; // rax
  _QWORD *v38; // rdx
  __int64 v39; // rax
  unsigned __int64 v40; // rsi
  unsigned __int64 v41; // rdx
  const __m128i *v42; // rcx
  unsigned __int64 v43; // r8
  __m128i *v44; // rax
  __int64 v45; // rdx
  __int64 v46; // rax
  _DWORD *v47; // rax
  __int64 v48; // r8
  __int64 v49; // rsi
  unsigned int v50; // edx
  __int64 v51; // rcx
  __int64 v52; // rdi
  __int64 v53; // r8
  __int64 (*v54)(); // r11
  char v55; // al
  __int64 v56; // r13
  __int64 j; // r9
  __int64 v58; // r8
  char v59; // al
  __int64 v60; // rax
  unsigned __int64 v61; // rdx
  _BYTE *v62; // rdi
  unsigned int v63; // ecx
  unsigned int v64; // eax
  __int64 v65; // rsi
  _DWORD *v66; // rdx
  __int64 v67; // rax
  int v68; // edx
  unsigned int v69; // r13d
  __int64 v70; // rdx
  unsigned int v71; // ecx
  int v72; // eax
  __int64 v73; // rax
  __int64 v74; // rdx
  __int64 v75; // rsi
  unsigned int v76; // ecx
  __int16 *v77; // rsi
  unsigned int v78; // edi
  unsigned int v79; // eax
  __int64 v80; // r10
  _DWORD *v81; // rdx
  __int64 v82; // rax
  _DWORD *v83; // rax
  int v84; // edx
  __int64 v85; // rdi
  const void *v86; // rsi
  __int16 *v87; // r8
  __int64 v88; // [rsp+8h] [rbp-98h]
  __int16 *v89; // [rsp+10h] [rbp-90h]
  int v90; // [rsp+10h] [rbp-90h]
  __int64 v91; // [rsp+20h] [rbp-80h]
  int v92; // [rsp+20h] [rbp-80h]
  int v93; // [rsp+28h] [rbp-78h]
  char *v94; // [rsp+28h] [rbp-78h]
  int v95; // [rsp+28h] [rbp-78h]
  __int64 v96; // [rsp+30h] [rbp-70h]
  __int64 v97; // [rsp+30h] [rbp-70h]
  __int64 v98; // [rsp+38h] [rbp-68h]
  __int64 *v99; // [rsp+40h] [rbp-60h] BYREF
  __int64 v100; // [rsp+48h] [rbp-58h]
  __int64 v101; // [rsp+50h] [rbp-50h] BYREF
  int v102; // [rsp+58h] [rbp-48h]

  v3 = a3;
  v5 = *(_QWORD **)(a2 + 112);
  v6 = *v5;
  v7 = v5[1];
  if ( *(_DWORD *)(*v5 + 72LL) != 1 )
  {
    if ( *(_DWORD *)(v7 + 72) != 1 )
      return 0;
    v6 = v5[1];
    v7 = *v5;
  }
  if ( *(_DWORD *)(v6 + 120) != 1 )
    return 0;
  v10 = **(_QWORD ***)(v6 + 112);
  a1[4] = v10;
  if ( v10 != (_QWORD *)v7
    && (*(_DWORD *)(v7 + 72) != 1 || *(_DWORD *)(v7 + 120) != 1 || v10 != **(_QWORD ***)(v7 + 112) || v10[24] != v10[23]) )
  {
    return 0;
  }
  if ( !(_BYTE)a3 )
  {
    if ( v10 + 6 == (_QWORD *)(v10[6] & 0xFFFFFFFFFFFFFFF8LL) )
      return 0;
    v11 = *(unsigned __int16 *)(v10[7] + 68LL);
    if ( v11 )
    {
      if ( v11 != 68 )
        return 0;
    }
  }
  *((_DWORD *)a1 + 84) = 0;
  v12 = *a1;
  v13 = *(__int64 (**)())(*(_QWORD *)*a1 + 344LL);
  if ( v13 == sub_2DB1AE0 )
    return 0;
  v8 = ((__int64 (__fastcall *)(__int64, __int64, _QWORD *, _QWORD *, _QWORD *, _QWORD))v13)(
         v12,
         a2,
         a1 + 5,
         a1 + 6,
         a1 + 41,
         0);
  if ( (_BYTE)v8 )
    return 0;
  v17 = a1[5];
  if ( !v17 || !*((_DWORD *)a1 + 84) )
    return 0;
  v18 = a1[4];
  *((_DWORD *)a1 + 16) = 0;
  if ( v17 == v6 )
    v6 = v7;
  a1[6] = v6;
  if ( v17 == v18 )
    v17 = a1[3];
  if ( v6 == v18 )
    v6 = a1[3];
  v19 = *(_QWORD *)(v18 + 56);
  v96 = v18 + 48;
  if ( v19 != v18 + 48 )
  {
    v20 = v3;
    v21 = v6;
    v22 = v17;
    v23 = v20;
    while ( *(_WORD *)(v19 + 68) == 68 || !*(_WORD *)(v19 + 68) )
    {
      v39 = *((unsigned int *)a1 + 16);
      v40 = *((unsigned int *)a1 + 17);
      v99 = (__int64 *)v19;
      v41 = a1[7];
      v42 = (const __m128i *)&v99;
      v100 = 0;
      v43 = v39 + 1;
      v101 = 0;
      v102 = 0;
      if ( v39 + 1 > v40 )
      {
        v85 = (__int64)(a1 + 7);
        v86 = a1 + 9;
        if ( v41 > (unsigned __int64)&v99 || (unsigned __int64)&v99 >= v41 + 32 * v39 )
        {
          v95 = v23;
          sub_C8D5F0(v85, v86, v43, 0x20u, v43, v16);
          v41 = a1[7];
          v39 = *((unsigned int *)a1 + 16);
          v42 = (const __m128i *)&v99;
          v23 = v95;
        }
        else
        {
          v92 = v23;
          v94 = (char *)&v99 - v41;
          sub_C8D5F0(v85, v86, v43, 0x20u, v43, v16);
          v41 = a1[7];
          v39 = *((unsigned int *)a1 + 16);
          v23 = v92;
          v42 = (const __m128i *)&v94[v41];
        }
      }
      v44 = (__m128i *)(v41 + 32 * v39);
      *v44 = _mm_loadu_si128(v42);
      v44[1] = _mm_loadu_si128(v42 + 1);
      v45 = a1[7];
      v46 = (unsigned int)(*((_DWORD *)a1 + 16) + 1);
      *((_DWORD *)a1 + 16) = v46;
      v47 = (_DWORD *)(v45 + 32 * v46 - 32);
      v48 = *(_QWORD *)v47;
      v49 = *(_QWORD *)(*(_QWORD *)v47 + 32LL);
      if ( (*(_DWORD *)(*(_QWORD *)v47 + 40LL) & 0xFFFFFF) != 1 )
      {
        v50 = 1;
        do
        {
          v51 = 40LL * (v50 + 1);
          v52 = *(_QWORD *)(v49 + v51 + 24);
          if ( v52 == v22 )
          {
            v47[2] = *(_DWORD *)(v49 + 40LL * v50 + 8);
            v49 = *(_QWORD *)(v48 + 32);
            v52 = *(_QWORD *)(v49 + v51 + 24);
          }
          if ( v21 == v52 )
          {
            v47[3] = *(_DWORD *)(v49 + 40LL * v50 + 8);
            v49 = *(_QWORD *)(v48 + 32);
          }
          v50 += 2;
        }
        while ( (*(_DWORD *)(v48 + 40) & 0xFFFFFF) != v50 );
      }
      v53 = *(unsigned int *)(v49 + 8);
      v54 = *(__int64 (**)())(*(_QWORD *)*a1 + 464LL);
      if ( v54 == sub_2DB1B10 )
        return 0;
      v93 = v23;
      a2 = a1[3];
      v55 = ((__int64 (__fastcall *)(_QWORD, __int64, _QWORD, _QWORD, __int64, _QWORD, _QWORD, _DWORD *, _DWORD *, _DWORD *))v54)(
              *a1,
              a2,
              a1[41],
              *((unsigned int *)a1 + 84),
              v53,
              (unsigned int)v47[2],
              (unsigned int)v47[3],
              v47 + 4,
              v47 + 5,
              v47 + 6);
      v23 = v93;
      if ( !v55 )
        return 0;
      if ( (*(_BYTE *)v19 & 4) == 0 )
      {
        while ( (*(_BYTE *)(v19 + 44) & 8) != 0 )
          v19 = *(_QWORD *)(v19 + 8);
      }
      v19 = *(_QWORD *)(v19 + 8);
      if ( v96 == v19 )
        break;
    }
    LOBYTE(v3) = v23;
  }
  ++a1[63];
  v98 = (__int64)(a1 + 63);
  if ( *((_BYTE *)a1 + 532) )
    goto LABEL_36;
  v24 = 4 * (*((_DWORD *)a1 + 131) - *((_DWORD *)a1 + 132));
  v25 = *((unsigned int *)a1 + 130);
  if ( v24 < 0x20 )
    v24 = 32;
  if ( v24 >= (unsigned int)v25 )
  {
    memset((void *)a1[64], -1, 8 * v25);
LABEL_36:
    *(_QWORD *)((char *)a1 + 524) = 0;
    goto LABEL_37;
  }
  sub_C8C990(v98, a2);
LABEL_37:
  v26 = 8LL * *((unsigned int *)a1 + 152);
  if ( v26 )
    memset((void *)a1[75], 0, v26);
  v27 = a1[5];
  v28 = a1[4];
  if ( (_BYTE)v3 )
  {
    if ( v28 != v27 )
    {
      if ( *(_QWORD *)(v27 + 192) != *(_QWORD *)(v27 + 184)
        || !(unsigned __int8)sub_2DB4750((__int64 **)a1, v27, v26, v14, v15) )
      {
        return 0;
      }
      v27 = a1[4];
    }
    v29 = a1[6];
    if ( v29 == v27 )
      goto LABEL_47;
    if ( *(_QWORD *)(v29 + 192) == *(_QWORD *)(v29 + 184) )
    {
      v27 = a1[6];
      if ( (unsigned __int8)sub_2DB4750((__int64 **)a1, v27, v26, v14, v29) )
        goto LABEL_47;
    }
    return 0;
  }
  if ( v28 != v27 )
  {
    if ( *(_QWORD *)(v27 + 192) != *(_QWORD *)(v27 + 184)
      || !(unsigned __int8)sub_2DB48E0((__int64)a1, v27, v26, v14, v15) )
    {
      return 0;
    }
    v27 = a1[4];
  }
  v29 = a1[6];
  if ( v29 != v27 )
  {
    if ( *(_QWORD *)(v29 + 192) != *(_QWORD *)(v29 + 184) )
      return 0;
    v27 = a1[6];
    if ( !(unsigned __int8)sub_2DB48E0((__int64)a1, v27, v26, v14, v29) )
      return 0;
  }
LABEL_47:
  v30 = a1[3];
  *((_DWORD *)a1 + 170) = 0;
  v99 = &v101;
  v100 = 0x800000000LL;
  v97 = sub_2E313E0(v30, v27, v26, v14, v29);
  v31 = a1[3];
  v32 = v31 + 48;
  v91 = *(_QWORD *)(v31 + 56);
  if ( v31 + 48 == v91 )
    goto LABEL_59;
  while ( 1 )
  {
    v33 = (_QWORD *)(*(_QWORD *)v32 & 0xFFFFFFFFFFFFFFF8LL);
    v34 = v33;
    if ( !v33 )
      BUG();
    v32 = *(_QWORD *)v32 & 0xFFFFFFFFFFFFFFF8LL;
    v35 = *v33;
    if ( (v35 & 4) == 0 && (*((_BYTE *)v34 + 44) & 4) != 0 )
    {
      for ( i = v35; ; i = *(_QWORD *)v32 )
      {
        v32 = i & 0xFFFFFFFFFFFFFFF8LL;
        if ( (*(_BYTE *)(v32 + 44) & 4) == 0 )
          break;
      }
    }
    if ( *((_BYTE *)a1 + 532) )
      break;
    if ( sub_C8CA60(v98, v32) )
      goto LABEL_59;
LABEL_86:
    v56 = *(_QWORD *)(v32 + 32);
    for ( j = v56 + 40LL * (*(_DWORD *)(v32 + 40) & 0xFFFFFF); j != v56; v56 += 40 )
    {
      if ( !*(_BYTE *)v56 )
      {
        v58 = *(unsigned int *)(v56 + 8);
        if ( (unsigned int)(v58 - 1) <= 0x3FFFFFFE )
        {
          if ( (*(_BYTE *)(v56 + 3) & 0x10) != 0 )
          {
            v74 = a1[1];
            v75 = *(_QWORD *)(v74 + 8);
            v76 = *(_DWORD *)(v75 + 24LL * (unsigned int)v58 + 16) & 0xFFF;
            v77 = (__int16 *)(*(_QWORD *)(v74 + 56) + 2LL * (*(_DWORD *)(v75 + 24LL * (unsigned int)v58 + 16) >> 12));
            do
            {
              if ( !v77 )
                break;
              v78 = *((_DWORD *)a1 + 170);
              v79 = *(unsigned __int8 *)(a1[90] + v76);
              if ( v79 < v78 )
              {
                v80 = a1[84];
                while ( 1 )
                {
                  v81 = (_DWORD *)(v80 + 4LL * v79);
                  if ( v76 == *v81 )
                    break;
                  v79 += 256;
                  if ( v78 <= v79 )
                    goto LABEL_132;
                }
                v82 = 4LL * v78;
                if ( v81 != (_DWORD *)(v80 + v82) )
                {
                  v83 = (_DWORD *)(v80 + v82 - 4);
                  if ( v81 != v83 )
                  {
                    *v81 = *v83;
                    *(_BYTE *)(a1[90] + *(unsigned int *)(a1[84] + 4LL * *((unsigned int *)a1 + 170) - 4)) = ((__int64)v81 - a1[84]) >> 2;
                    v78 = *((_DWORD *)a1 + 170);
                  }
                  *((_DWORD *)a1 + 170) = v78 - 1;
                }
              }
LABEL_132:
              v84 = *v77++;
              v76 += v84;
            }
            while ( (_WORD)v84 );
          }
          v59 = *(_BYTE *)(v56 + 4);
          if ( (v59 & 1) == 0
            && (v59 & 2) == 0
            && ((*(_BYTE *)(v56 + 3) & 0x10) == 0 || (*(_DWORD *)v56 & 0xFFF00) != 0) )
          {
            v60 = (unsigned int)v100;
            v61 = (unsigned int)v100 + 1LL;
            if ( v61 > HIDWORD(v100) )
            {
              v88 = j;
              v90 = v58;
              sub_C8D5F0((__int64)&v99, &v101, v61, 4u, v58, j);
              v60 = (unsigned int)v100;
              j = v88;
              LODWORD(v58) = v90;
            }
            *((_DWORD *)v99 + v60) = v58;
            LODWORD(v100) = v100 + 1;
          }
        }
      }
    }
    while ( (_DWORD)v100 )
    {
      v70 = a1[1];
      v71 = *((_DWORD *)v99 + (unsigned int)v100 - 1);
      LODWORD(v100) = v100 - 1;
      v69 = *(_DWORD *)(*(_QWORD *)(v70 + 8) + 24LL * v71 + 16) & 0xFFF;
      v87 = (__int16 *)(*(_QWORD *)(v70 + 56) + 2LL * (*(_DWORD *)(*(_QWORD *)(v70 + 8) + 24LL * v71 + 16) >> 12));
      do
      {
        if ( !v87 )
          break;
        if ( (*(_QWORD *)(a1[75] + 8LL * (v69 >> 6)) & (1LL << v69)) != 0 )
        {
          v62 = (_BYTE *)(a1[90] + v69);
          v63 = *((_DWORD *)a1 + 170);
          v64 = (unsigned __int8)*v62;
          if ( v64 >= v63 )
            goto LABEL_104;
          v65 = a1[84];
          while ( 1 )
          {
            v66 = (_DWORD *)(v65 + 4LL * v64);
            if ( v69 == *v66 )
              break;
            v64 += 256;
            if ( v63 <= v64 )
              goto LABEL_104;
          }
          if ( v66 == (_DWORD *)(v65 + 4LL * v63) )
          {
LABEL_104:
            *v62 = v63;
            v67 = *((unsigned int *)a1 + 170);
            if ( v67 + 1 > (unsigned __int64)*((unsigned int *)a1 + 171) )
            {
              v89 = v87;
              sub_C8D5F0((__int64)(a1 + 84), a1 + 86, v67 + 1, 4u, (__int64)v87, j);
              v67 = *((unsigned int *)a1 + 170);
              v87 = v89;
            }
            *(_DWORD *)(a1[84] + 4 * v67) = v69;
            ++*((_DWORD *)a1 + 170);
          }
        }
        v68 = *v87++;
        v69 += v68;
      }
      while ( (_WORD)v68 );
    }
    if ( v32 == v97
      || ((v72 = *(_DWORD *)(v32 + 44), (v72 & 4) != 0) || (v72 & 8) == 0
        ? (v73 = (*(_QWORD *)(*(_QWORD *)(v32 + 16) + 24LL) >> 9) & 1LL)
        : (LOBYTE(v73) = sub_2E88A90(v32, 512, 1)),
          !(_BYTE)v73) )
    {
      if ( !*((_DWORD *)a1 + 170) )
      {
        a1[92] = v32;
        if ( v99 != &v101 )
          _libc_free((unsigned __int64)v99);
        return 1;
      }
    }
    if ( v91 == v32 )
      goto LABEL_59;
  }
  v37 = (_QWORD *)a1[64];
  v38 = &v37[*((unsigned int *)a1 + 131)];
  if ( v37 == v38 )
    goto LABEL_86;
  while ( *v37 != v32 )
  {
    if ( v38 == ++v37 )
      goto LABEL_86;
  }
LABEL_59:
  if ( v99 != &v101 )
    _libc_free((unsigned __int64)v99);
  return v8;
}
