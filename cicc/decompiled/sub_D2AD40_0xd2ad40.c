// Function: sub_D2AD40
// Address: 0xd2ad40
//
__int64 __fastcall sub_D2AD40(__int64 a1, __int64 *a2)
{
  __int64 result; // rax
  __int64 v3; // rcx
  _QWORD *v5; // rax
  _QWORD *v6; // rdx
  _QWORD *v7; // r13
  _QWORD *v8; // rbx
  __int64 v9; // rax
  __int64 v10; // rcx
  __int64 v11; // r9
  unsigned __int64 v12; // r8
  __int64 v13; // rdx
  _QWORD *v14; // r15
  __int64 v15; // rax
  unsigned __int64 v16; // rdx
  unsigned __int64 v17; // r12
  _BYTE *v18; // r12
  __int64 v19; // rbx
  __int64 v20; // rdx
  bool v21; // zf
  __int64 *v22; // rdx
  __int64 *v23; // rsi
  __int64 v24; // rdi
  __int64 v25; // rax
  int v26; // ecx
  __int64 v27; // rdx
  __int64 *v28; // rax
  int v29; // r13d
  unsigned __int64 *v30; // rdx
  unsigned __int64 v31; // r12
  __int64 *v32; // rax
  __int64 i; // rcx
  unsigned __int64 v34; // rbx
  int v35; // edx
  int v36; // edx
  __int64 v37; // rax
  __int64 v38; // rdx
  __int64 v39; // rcx
  __int64 v40; // r8
  int v41; // edx
  unsigned __int64 *v42; // rdi
  __int64 v43; // rdx
  __int64 v44; // rdx
  char *v45; // rbx
  __int64 v46; // rsi
  __int64 v47; // rdx
  char *v48; // r12
  __int64 v49; // rax
  _QWORD *v50; // r9
  __int64 v51; // r8
  __int64 v52; // rbx
  unsigned __int64 v53; // rdx
  const __m128i *v54; // r12
  __m128i *v55; // rax
  _QWORD *v56; // rax
  char *v57; // r12
  const __m128i *v58; // rbx
  __m128i *v59; // rax
  char *v60; // rbx
  _BYTE *v61; // [rsp-340h] [rbp-340h]
  int v62; // [rsp-32Ch] [rbp-32Ch]
  _BYTE *v63; // [rsp-328h] [rbp-328h]
  __int64 v64; // [rsp-320h] [rbp-320h]
  __int64 v65; // [rsp-310h] [rbp-310h] BYREF
  _QWORD *v66; // [rsp-308h] [rbp-308h] BYREF
  __int64 *v67; // [rsp-300h] [rbp-300h]
  __int64 *v68; // [rsp-2F8h] [rbp-2F8h]
  _BYTE *v69; // [rsp-2E8h] [rbp-2E8h] BYREF
  __int64 v70; // [rsp-2E0h] [rbp-2E0h]
  _BYTE v71[128]; // [rsp-2D8h] [rbp-2D8h] BYREF
  char *v72; // [rsp-258h] [rbp-258h] BYREF
  __int64 v73; // [rsp-250h] [rbp-250h]
  _BYTE v74[128]; // [rsp-248h] [rbp-248h] BYREF
  __int64 v75; // [rsp-1C8h] [rbp-1C8h] BYREF
  __int64 v76; // [rsp-1C0h] [rbp-1C0h]
  _BYTE v77[440]; // [rsp-1B8h] [rbp-1B8h] BYREF

  result = *(_QWORD *)(a1 + 128);
  v3 = result + 8LL * *(unsigned int *)(a1 + 136);
  if ( result == v3 )
    return result;
  while ( (*(_QWORD *)result & 0xFFFFFFFFFFFFFFF8LL) == 0 || !*(_QWORD *)(*(_QWORD *)result & 0xFFFFFFFFFFFFFFF8LL) )
  {
    result += 8;
    if ( v3 == result )
      return result;
  }
  result = *(unsigned int *)(a1 + 440);
  if ( (_DWORD)result )
    return result;
  v69 = v71;
  v70 = 0x1000000000LL;
  v5 = sub_D23BF0(a1 + 128);
  v7 = v6;
  v8 = v5;
  v9 = sub_D23C30(a1 + 128);
  v12 = v9;
  if ( (_QWORD *)v9 == v8 )
  {
    v13 = (unsigned int)v70;
  }
  else
  {
    LODWORD(v13) = v70;
    v14 = (_QWORD *)v9;
    do
    {
LABEL_12:
      v15 = (unsigned int)v13;
      v10 = HIDWORD(v70);
      v16 = (unsigned int)v13 + 1LL;
      v17 = *v8 & 0xFFFFFFFFFFFFFFF8LL;
      if ( v16 > HIDWORD(v70) )
      {
        a2 = (__int64 *)v71;
        sub_C8D5F0((__int64)&v69, v71, v16, 8u, v12, v11);
        v15 = (unsigned int)v70;
      }
      ++v8;
      *(_QWORD *)&v69[8 * v15] = v17;
      v13 = (unsigned int)(v70 + 1);
      LODWORD(v70) = v70 + 1;
      if ( v7 != v8 )
      {
        while ( (*v8 & 0xFFFFFFFFFFFFFFF8LL) == 0 || !*(_QWORD *)(*v8 & 0xFFFFFFFFFFFFFFF8LL) )
        {
          if ( v7 == ++v8 )
          {
            if ( v14 != v8 )
              goto LABEL_12;
            goto LABEL_18;
          }
        }
      }
    }
    while ( v14 != v8 );
  }
LABEL_18:
  v18 = v69;
  v75 = (__int64)v77;
  v76 = 0x1000000000LL;
  v73 = 0x1000000000LL;
  result = (__int64)&v69[8 * v13];
  v72 = v74;
  v63 = (_BYTE *)result;
  if ( v69 == (_BYTE *)result )
    goto LABEL_83;
  result = (__int64)&v66;
  do
  {
    while ( 1 )
    {
      v19 = *(_QWORD *)v18;
      v20 = *(unsigned int *)(*(_QWORD *)v18 + 16LL);
      if ( !(_DWORD)v20 )
        break;
      v18 += 8;
      if ( v63 == v18 )
        goto LABEL_78;
    }
    v21 = *(_BYTE *)(v19 + 104) == 0;
    *(_QWORD *)(v19 + 16) = 0x100000001LL;
    if ( v21 )
      sub_D29180(v19, (__int64)a2, v20, v10, v12, v11);
    v23 = sub_D23BF0(v19 + 24);
    v24 = (__int64)v22;
    v25 = (unsigned int)v76;
    v26 = v76;
    if ( (unsigned int)v76 >= (unsigned __int64)HIDWORD(v76) )
    {
      v12 = (unsigned int)v76 + 1LL;
      v66 = (_QWORD *)v19;
      v10 = v75;
      v67 = v23;
      v58 = (const __m128i *)&v66;
      v68 = v22;
      if ( HIDWORD(v76) < v12 )
      {
        if ( v75 > (unsigned __int64)&v66 || (unsigned __int64)&v66 >= v75 + 24 * (unsigned __int64)(unsigned int)v76 )
        {
          sub_C8D5F0((__int64)&v75, v77, v12, 0x18u, v12, v11);
          v10 = v75;
          v25 = (unsigned int)v76;
        }
        else
        {
          v60 = (char *)&v66 - v75;
          sub_C8D5F0((__int64)&v75, v77, v12, 0x18u, v12, v11);
          v10 = v75;
          v25 = (unsigned int)v76;
          v58 = (const __m128i *)&v60[v75];
        }
      }
      v59 = (__m128i *)(v10 + 24 * v25);
      *v59 = _mm_loadu_si128(v58);
      v59[1].m128i_i64[0] = v58[1].m128i_i64[0];
      v27 = v75;
      LODWORD(v10) = v76 + 1;
      LODWORD(v76) = v76 + 1;
    }
    else
    {
      v27 = v75;
      v28 = (__int64 *)(v75 + 24LL * (unsigned int)v76);
      if ( v28 )
      {
        v28[2] = v24;
        *v28 = v19;
        v27 = v75;
        v28[1] = (__int64)v23;
        v26 = v76;
      }
      LODWORD(v10) = v26 + 1;
      LODWORD(v76) = v10;
    }
    v61 = v18;
    v29 = 2;
    while ( 1 )
    {
      v30 = (unsigned __int64 *)(v27 + 24LL * (unsigned int)v10 - 24);
      v31 = *v30;
      v32 = (__int64 *)v30[1];
      LODWORD(v76) = v10 - 1;
      a2 = (__int64 *)v30[2];
      i = *(_QWORD *)(v31 + 24) + 8LL * *(unsigned int *)(v31 + 32);
      while ( (__int64 *)i != v32 )
      {
LABEL_33:
        v34 = *v32 & 0xFFFFFFFFFFFFFFF8LL;
        v35 = *(_DWORD *)(v34 + 16);
        if ( v35 )
        {
          ++v32;
          if ( v35 == -1 )
          {
            if ( v32 != a2 )
            {
              while ( (*v32 & 0xFFFFFFFFFFFFFFF8LL) == 0 || !*(_QWORD *)(*v32 & 0xFFFFFFFFFFFFFFF8LL) )
              {
                if ( ++v32 == a2 )
                {
                  if ( (__int64 *)i != v32 )
                    goto LABEL_33;
                  goto LABEL_45;
                }
              }
            }
          }
          else
          {
            v36 = *(_DWORD *)(v34 + 20);
            if ( v36 < *(_DWORD *)(v31 + 20) )
              *(_DWORD *)(v31 + 20) = v36;
            while ( v32 != a2 && ((*v32 & 0xFFFFFFFFFFFFFFF8LL) == 0 || !*(_QWORD *)(*v32 & 0xFFFFFFFFFFFFFFF8LL)) )
              ++v32;
          }
        }
        else
        {
          v39 = (unsigned int)v76;
          v40 = v75;
          v11 = HIDWORD(v76);
          v41 = v76;
          v42 = (unsigned __int64 *)(v75 + 24LL * (unsigned int)v76);
          if ( (unsigned int)v76 >= (unsigned __int64)HIDWORD(v76) )
          {
            v53 = (unsigned int)v76 + 1LL;
            v66 = (_QWORD *)v31;
            v54 = (const __m128i *)&v66;
            v67 = v32;
            v68 = a2;
            if ( HIDWORD(v76) < v53 )
            {
              if ( v75 > (unsigned __int64)&v66 || v42 <= (unsigned __int64 *)&v66 )
              {
                a2 = (__int64 *)v77;
                v54 = (const __m128i *)&v66;
                sub_C8D5F0((__int64)&v75, v77, v53, 0x18u, v75, HIDWORD(v76));
                v40 = v75;
                v39 = (unsigned int)v76;
              }
              else
              {
                a2 = (__int64 *)v77;
                v57 = (char *)&v66 - v75;
                sub_C8D5F0((__int64)&v75, v77, v53, 0x18u, v75, HIDWORD(v76));
                v40 = v75;
                v39 = (unsigned int)v76;
                v54 = (const __m128i *)&v57[v75];
              }
            }
            v55 = (__m128i *)(v40 + 24 * v39);
            *v55 = _mm_loadu_si128(v54);
            v43 = v54[1].m128i_i64[0];
            LODWORD(v76) = v76 + 1;
            v55[1].m128i_i64[0] = v43;
          }
          else
          {
            if ( v42 )
            {
              v42[2] = (unsigned __int64)a2;
              *v42 = v31;
              v42[1] = (unsigned __int64)v32;
              v41 = v76;
            }
            v43 = (unsigned int)(v41 + 1);
            LODWORD(v76) = v43;
          }
          v21 = *(_BYTE *)(v34 + 104) == 0;
          *(_DWORD *)(v34 + 20) = v29;
          v12 = (unsigned int)(v29 + 1);
          *(_DWORD *)(v34 + 16) = v29;
          if ( v21 )
          {
            sub_D29180(v34, (__int64)a2, v43, v39, v12, v11);
            v12 = (unsigned int)(v29 + 1);
          }
          v32 = *(__int64 **)(v34 + 24);
          for ( i = (__int64)&v32[*(unsigned int *)(v34 + 32)]; (__int64 *)i != v32; ++v32 )
          {
            if ( (*v32 & 0xFFFFFFFFFFFFFFF8LL) != 0 && *(_QWORD *)(*v32 & 0xFFFFFFFFFFFFFFF8LL) )
              break;
          }
          a2 = (__int64 *)i;
          v31 = v34;
          v29 = v12;
        }
      }
LABEL_45:
      v37 = (unsigned int)v73;
      if ( (unsigned __int64)(unsigned int)v73 + 1 > HIDWORD(v73) )
      {
        a2 = (__int64 *)v74;
        sub_C8D5F0((__int64)&v72, v74, (unsigned int)v73 + 1LL, 8u, v12, v11);
        v37 = (unsigned int)v73;
      }
      *(_QWORD *)&v72[8 * v37] = v31;
      v38 = (unsigned int)(v73 + 1);
      result = *(unsigned int *)(v31 + 16);
      LODWORD(v73) = v73 + 1;
      if ( *(_DWORD *)(v31 + 20) == (_DWORD)result )
        break;
      v10 = (unsigned int)v76;
      if ( !(_DWORD)v76 )
        goto LABEL_77;
LABEL_49:
      v27 = v75;
    }
    v44 = 8 * v38;
    v45 = &v72[v44];
    v46 = v44 >> 3;
    v47 = v44 >> 5;
    if ( !v47 )
    {
      v48 = v45;
LABEL_92:
      if ( v46 != 2 )
      {
        if ( v46 != 3 )
        {
          if ( v46 != 1 )
          {
            v48 = v72;
            goto LABEL_69;
          }
          goto LABEL_100;
        }
        if ( (int)result > *(_DWORD *)(*((_QWORD *)v48 - 1) + 16LL) )
          goto LABEL_69;
        v48 -= 8;
      }
      if ( (int)result > *(_DWORD *)(*((_QWORD *)v48 - 1) + 16LL) )
        goto LABEL_69;
      v48 -= 8;
LABEL_100:
      if ( (int)result <= *(_DWORD *)(*((_QWORD *)v48 - 1) + 16LL) )
        v48 = v72;
      goto LABEL_69;
    }
    v48 = v45;
    while ( (int)result <= *(_DWORD *)(*((_QWORD *)v48 - 1) + 16LL) )
    {
      if ( (int)result > *(_DWORD *)(*((_QWORD *)v48 - 2) + 16LL) )
      {
        v48 -= 8;
        break;
      }
      if ( (int)result > *(_DWORD *)(*((_QWORD *)v48 - 3) + 16LL) )
      {
        v48 -= 16;
        break;
      }
      if ( (int)result > *(_DWORD *)(*((_QWORD *)v48 - 4) + 16LL) )
      {
        v48 -= 24;
        break;
      }
      v48 -= 32;
      if ( &v45[-32 * v47] == v48 )
      {
        v46 = (v48 - v72) >> 3;
        goto LABEL_92;
      }
    }
LABEL_69:
    v49 = *(_QWORD *)(a1 + 336);
    *(_QWORD *)(a1 + 416) += 136LL;
    v50 = (_QWORD *)((v49 + 7) & 0xFFFFFFFFFFFFFFF8LL);
    if ( *(_QWORD *)(a1 + 344) >= (unsigned __int64)(v50 + 17) && v49 )
      *(_QWORD *)(a1 + 336) = v50 + 17;
    else
      v50 = (_QWORD *)sub_9D1E70(a1 + 336, 136, 136, 3);
    v64 = (__int64)v50;
    sub_D23F30(v50, a1);
    v66 = v45;
    v65 = v64;
    v67 = (__int64 *)v48;
    sub_D2AB20(a1, v64, (__int64 *)&v66);
    v62 = *(_DWORD *)(a1 + 440);
    a2 = &v65;
    if ( !(unsigned __int8)sub_D24D10(a1 + 576, &v65, &v66) )
    {
      v56 = sub_D27750(a1 + 576, &v65, v66);
      *v56 = v65;
      *((_DWORD *)v56 + 2) = v62;
    }
    result = *(unsigned int *)(a1 + 440);
    v52 = v65;
    if ( result + 1 > (unsigned __int64)*(unsigned int *)(a1 + 444) )
    {
      a2 = (__int64 *)(a1 + 448);
      sub_C8D5F0(a1 + 432, (const void *)(a1 + 448), result + 1, 8u, v51, v11);
      result = *(unsigned int *)(a1 + 440);
    }
    *(_QWORD *)(*(_QWORD *)(a1 + 432) + 8 * result) = v52;
    v10 = (unsigned int)v76;
    v12 = (v48 - v72) >> 3;
    ++*(_DWORD *)(a1 + 440);
    LODWORD(v73) = v12;
    if ( (_DWORD)v10 )
      goto LABEL_49;
LABEL_77:
    v18 = v61 + 8;
  }
  while ( v63 != v61 + 8 );
LABEL_78:
  if ( v72 != v74 )
    result = _libc_free(v72, a2);
  if ( (_BYTE *)v75 != v77 )
    result = _libc_free(v75, a2);
  v18 = v69;
LABEL_83:
  if ( v18 != v71 )
    return _libc_free(v18, a2);
  return result;
}
