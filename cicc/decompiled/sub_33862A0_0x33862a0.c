// Function: sub_33862A0
// Address: 0x33862a0
//
_BYTE *__fastcall sub_33862A0(
        __int64 a1,
        __int64 a2,
        __int64 a3,
        __int64 a4,
        __int64 a5,
        __int64 a6,
        __int64 *a7,
        __int64 a8,
        _BYTE *a9)
{
  __int64 *v9; // r13
  _BYTE *result; // rax
  __int64 v15; // rdx
  __int64 v16; // rcx
  __int64 v17; // rsi
  unsigned int v18; // eax
  __int64 v19; // rdi
  unsigned int v20; // esi
  __int64 v21; // r15
  __int64 v22; // r9
  __int64 *v23; // rax
  __int64 v24; // rdi
  __int64 v25; // r9
  _DWORD *v26; // rdx
  __int64 v27; // r8
  int v28; // edi
  __int64 v29; // rcx
  __int64 v30; // rsi
  __int64 v31; // rdi
  unsigned __int64 v32; // rcx
  unsigned int v33; // esi
  __int64 v34; // rdi
  int v35; // r12d
  unsigned __int64 v36; // rcx
  __int64 v37; // rdx
  int v38; // eax
  unsigned __int64 v39; // r12
  __int64 v40; // rdi
  __int64 v41; // rbx
  unsigned __int64 v42; // r15
  _QWORD *v43; // rdx
  __int64 v44; // rbx
  _QWORD *v45; // rax
  _QWORD *v46; // rdx
  int v47; // eax
  int v48; // r8d
  _DWORD *v49; // r15
  int v50; // eax
  int v51; // r9d
  int v52; // edi
  int v53; // edi
  __int64 v54; // rsi
  unsigned int v55; // r9d
  __int64 v56; // rcx
  int v57; // eax
  _QWORD *v58; // r10
  int v59; // r9d
  int v60; // r9d
  __int64 v61; // rsi
  int v62; // eax
  __int64 v63; // rdi
  __int64 v64; // rcx
  int v65; // eax
  int v66; // r12d
  __int64 v67; // r10
  unsigned int v68; // edi
  int v69; // esi
  _DWORD *v70; // rax
  int v71; // r10d
  int v72; // r10d
  int v73; // edi
  unsigned int v74; // r12d
  _DWORD *v75; // rsi
  int v76; // eax
  unsigned int v77; // edi
  unsigned int v78; // [rsp+4h] [rbp-6Ch]
  int v79; // [rsp+8h] [rbp-68h]
  __int64 v80; // [rsp+8h] [rbp-68h]
  unsigned int v81; // [rsp+8h] [rbp-68h]
  unsigned int v82; // [rsp+8h] [rbp-68h]
  unsigned __int64 v83; // [rsp+10h] [rbp-60h]
  unsigned int v84; // [rsp+18h] [rbp-58h]
  int v85; // [rsp+18h] [rbp-58h]
  __int64 v86; // [rsp+18h] [rbp-58h]
  __int64 v87; // [rsp+18h] [rbp-58h]
  __int64 v88; // [rsp+18h] [rbp-58h]
  int v89; // [rsp+20h] [rbp-50h]
  __int64 *v90; // [rsp+28h] [rbp-48h]

  v9 = a7;
  result = (_BYTE *)*a7;
  if ( *(_DWORD *)(*a7 + 24) != 298 )
    return result;
  v15 = *(_QWORD *)(*((_QWORD *)result + 5) + 40LL);
  result = (_BYTE *)*(unsigned int *)(v15 + 24);
  if ( (_DWORD)result != 15 && (_DWORD)result != 39 )
    return result;
  v16 = *(unsigned int *)(a5 + 24);
  v17 = *(_QWORD *)(a5 + 8);
  if ( (_DWORD)v16 )
  {
    v18 = (v16 - 1) & (((unsigned int)a6 >> 9) ^ ((unsigned int)a6 >> 4));
    v90 = (__int64 *)(v17 + 24LL * v18);
    v19 = *v90;
    if ( a6 == *v90 )
      goto LABEL_7;
    v51 = 1;
    while ( v19 != -4096 )
    {
      v18 = (v16 - 1) & (v51 + v18);
      v19 = *(_QWORD *)(v17 + 24LL * v18);
      if ( a6 == v19 )
      {
        v90 = (__int64 *)(v17 + 24LL * v18);
        goto LABEL_7;
      }
      ++v51;
    }
  }
  v90 = (__int64 *)(v17 + 24 * v16);
LABEL_7:
  v20 = *(_DWORD *)(a1 + 272);
  v21 = v90[1];
  v89 = *(_DWORD *)(v15 + 96);
  if ( !v20 )
  {
    ++*(_QWORD *)(a1 + 248);
    goto LABEL_56;
  }
  v22 = *(_QWORD *)(a1 + 256);
  v84 = (v20 - 1) & (((unsigned int)v21 >> 9) ^ ((unsigned int)v21 >> 4));
  v23 = (__int64 *)(v22 + 16LL * v84);
  v24 = *v23;
  if ( v21 == *v23 )
  {
LABEL_9:
    v25 = *((unsigned int *)v23 + 2);
    v26 = v23 + 1;
    v27 = v25;
    goto LABEL_10;
  }
  v79 = 1;
  v46 = 0;
  while ( v24 != -4096 )
  {
    if ( v24 == -8192 && !v46 )
      v46 = v23;
    v84 = (v20 - 1) & (v84 + v79);
    v23 = (__int64 *)(v22 + 16LL * v84);
    v24 = *v23;
    if ( v21 == *v23 )
      goto LABEL_9;
    ++v79;
  }
  if ( !v46 )
    v46 = v23;
  v47 = *(_DWORD *)(a1 + 264);
  ++*(_QWORD *)(a1 + 248);
  v48 = v47 + 1;
  if ( 4 * (v47 + 1) >= 3 * v20 )
  {
LABEL_56:
    v86 = a3;
    sub_3383550(a1 + 248, 2 * v20);
    v52 = *(_DWORD *)(a1 + 272);
    if ( v52 )
    {
      v53 = v52 - 1;
      a3 = v86;
      v54 = *(_QWORD *)(a1 + 256);
      v55 = v53 & (((unsigned int)v21 >> 9) ^ ((unsigned int)v21 >> 4));
      v48 = *(_DWORD *)(a1 + 264) + 1;
      v46 = (_QWORD *)(v54 + 16LL * v55);
      v56 = *v46;
      if ( v21 == *v46 )
        goto LABEL_38;
      v57 = 1;
      v58 = 0;
      while ( v56 != -4096 )
      {
        if ( v56 == -8192 && !v58 )
          v58 = v46;
        v55 = v53 & (v57 + v55);
        v46 = (_QWORD *)(v54 + 16LL * v55);
        v56 = *v46;
        if ( v21 == *v46 )
          goto LABEL_38;
        ++v57;
      }
LABEL_60:
      if ( v58 )
        v46 = v58;
      goto LABEL_38;
    }
LABEL_111:
    ++*(_DWORD *)(a1 + 264);
    BUG();
  }
  if ( v20 - *(_DWORD *)(a1 + 268) - v48 <= v20 >> 3 )
  {
    v80 = a3;
    sub_3383550(a1 + 248, v20);
    v59 = *(_DWORD *)(a1 + 272);
    if ( v59 )
    {
      v60 = v59 - 1;
      v58 = 0;
      a3 = v80;
      v61 = *(_QWORD *)(a1 + 256);
      v48 = *(_DWORD *)(a1 + 264) + 1;
      v62 = 1;
      v63 = v60 & (((unsigned int)v21 >> 9) ^ ((unsigned int)v21 >> 4));
      v46 = (_QWORD *)(v61 + 16 * v63);
      v64 = *v46;
      if ( v21 == *v46 )
        goto LABEL_38;
      while ( v64 != -4096 )
      {
        if ( v64 == -8192 && !v58 )
          v58 = v46;
        LODWORD(v63) = v60 & (v62 + v63);
        v46 = (_QWORD *)(v61 + 16LL * (unsigned int)v63);
        v64 = *v46;
        if ( v21 == *v46 )
          goto LABEL_38;
        ++v62;
      }
      goto LABEL_60;
    }
    goto LABEL_111;
  }
LABEL_38:
  *(_DWORD *)(a1 + 264) = v48;
  if ( *v46 != -4096 )
    --*(_DWORD *)(a1 + 268);
  *v46 = v21;
  v27 = 0;
  v26 = v46 + 1;
  v25 = 0;
  *v26 = 0;
LABEL_10:
  result = *(_BYTE **)(*(_QWORD *)(a1 + 8) + 48LL);
  v28 = *((_DWORD *)result + 8);
  v29 = *((_QWORD *)result + 1);
  v30 = v29 + 40LL * (unsigned int)(v28 + v89);
  v31 = v29 + 40LL * (unsigned int)(v27 + v28);
  if ( *(_QWORD *)(v31 + 8) != *(_QWORD *)(v30 + 8) )
    return result;
  _BitScanReverse64(&v32, 1LL << *(_WORD *)(v21 + 2));
  if ( *(_BYTE *)(v30 + 16) < (unsigned __int8)(63 - (v32 ^ 0x3F)) )
    return result;
  *(_QWORD *)(v31 + 8) = -1;
  *(_BYTE *)(*((_QWORD *)result + 1) + 40LL * (unsigned int)(*((_DWORD *)result + 8) + v89) + 17) = 0;
  *(_BYTE *)(*((_QWORD *)result + 1) + 40LL * (unsigned int)(*((_DWORD *)result + 8) + v89) + 33) = 1;
  *v26 = v89;
  v33 = *(_DWORD *)(a3 + 24);
  if ( !v33 )
  {
    ++*(_QWORD *)a3;
    goto LABEL_72;
  }
  v34 = *(_QWORD *)(a3 + 8);
  v35 = 37 * v27;
  v36 = (v33 - 1) & (37 * (_DWORD)v27);
  v37 = v34 + 8 * v36;
  v38 = *(_DWORD *)v37;
  if ( *(_DWORD *)v37 != (_DWORD)v25 )
  {
    v85 = 1;
    v49 = 0;
    while ( v38 != 0x7FFFFFFF )
    {
      if ( v38 == 0x80000000 && !v49 )
        v49 = (_DWORD *)v37;
      v36 = (v33 - 1) & (v85 + (_DWORD)v36);
      v37 = v34 + 8LL * (unsigned int)v36;
      v38 = *(_DWORD *)v37;
      if ( *(_DWORD *)v37 == (_DWORD)v25 )
        goto LABEL_14;
      ++v85;
    }
    v50 = *(_DWORD *)(a3 + 16);
    if ( !v49 )
      v49 = (_DWORD *)v37;
    ++*(_QWORD *)a3;
    v36 = (unsigned int)(v50 + 1);
    if ( 4 * (int)v36 < 3 * v33 )
    {
      v37 = v33 >> 3;
      if ( v33 - *(_DWORD *)(a3 + 20) - (unsigned int)v36 > (unsigned int)v37 )
      {
LABEL_47:
        *(_DWORD *)(a3 + 16) = v36;
        if ( *v49 != 0x7FFFFFFF )
          --*(_DWORD *)(a3 + 20);
        *v49 = v25;
        v49[1] = v89;
        goto LABEL_14;
      }
      v88 = a3;
      v82 = v25;
      sub_2FC3A50(a3, v33);
      a3 = v88;
      v71 = *(_DWORD *)(v88 + 24);
      if ( v71 )
      {
        v72 = v71 - 1;
        v27 = *(_QWORD *)(v88 + 8);
        v73 = 1;
        v74 = v72 & v35;
        v25 = v82;
        v36 = (unsigned int)(*(_DWORD *)(v88 + 16) + 1);
        v75 = 0;
        v49 = (_DWORD *)(v27 + 8LL * v74);
        v76 = *v49;
        if ( *v49 != v82 )
        {
          while ( v76 != 0x7FFFFFFF )
          {
            if ( v76 == 0x80000000 && !v75 )
              v75 = v49;
            v37 = (unsigned int)(v73 + 1);
            v77 = v72 & (v74 + v73);
            v49 = (_DWORD *)(v27 + 8LL * v77);
            v74 = v77;
            v76 = *v49;
            if ( *v49 == v82 )
              goto LABEL_47;
            v73 = v37;
          }
          if ( v75 )
            v49 = v75;
        }
        goto LABEL_47;
      }
LABEL_112:
      ++*(_DWORD *)(a3 + 16);
      BUG();
    }
LABEL_72:
    v87 = a3;
    v78 = v25;
    v81 = v27;
    sub_2FC3A50(a3, 2 * v33);
    a3 = v87;
    v65 = *(_DWORD *)(v87 + 24);
    if ( v65 )
    {
      v27 = v81;
      v66 = v65 - 1;
      v67 = *(_QWORD *)(v87 + 8);
      v25 = v78;
      v68 = (v65 - 1) & (37 * v81);
      v49 = (_DWORD *)(v67 + 8LL * v68);
      v69 = *v49;
      v36 = (unsigned int)(*(_DWORD *)(v87 + 16) + 1);
      if ( *v49 != v78 )
      {
        v27 = 1;
        v70 = 0;
        while ( v69 != 0x7FFFFFFF )
        {
          if ( !v70 && v69 == 0x80000000 )
            v70 = v49;
          v37 = (unsigned int)(v27 + 1);
          v27 = v68 + (unsigned int)v27;
          v68 = v66 & v27;
          v49 = (_DWORD *)(v67 + 8LL * (v66 & (unsigned int)v27));
          v69 = *v49;
          if ( *v49 == v78 )
            goto LABEL_47;
          v27 = (unsigned int)v37;
        }
        if ( v70 )
          v49 = v70;
      }
      goto LABEL_47;
    }
    goto LABEL_112;
  }
LABEL_14:
  if ( a7 != &a7[2 * a8] )
  {
    v37 = *(unsigned int *)(a2 + 8);
    v39 = v83;
    v40 = a2;
    v27 = 0xFFFFFFFF00000000LL;
    do
    {
      v36 = *(unsigned int *)(v40 + 12);
      v25 = v37 + 1;
      v41 = *v9;
      v42 = v39 & 0xFFFFFFFF00000000LL | 1;
      v39 = v42;
      if ( v37 + 1 > v36 )
      {
        sub_C8D5F0(v40, (const void *)(v40 + 16), v37 + 1, 0x10u, 0xFFFFFFFF00000000LL, v25);
        v27 = 0xFFFFFFFF00000000LL;
        v37 = *(unsigned int *)(v40 + 8);
      }
      v43 = (_QWORD *)(*(_QWORD *)v40 + 16 * v37);
      v9 += 2;
      *v43 = v41;
      v43[1] = v42;
      v37 = (unsigned int)(*(_DWORD *)(v40 + 8) + 1);
      *(_DWORD *)(v40 + 8) = v37;
    }
    while ( &a7[2 * a8] != v9 );
  }
  v44 = v90[2];
  if ( !*(_BYTE *)(a4 + 28) )
    goto LABEL_50;
  v45 = *(_QWORD **)(a4 + 8);
  v36 = *(unsigned int *)(a4 + 20);
  v37 = (__int64)&v45[v36];
  if ( v45 != (_QWORD *)v37 )
  {
    while ( v44 != *v45 )
    {
      if ( (_QWORD *)v37 == ++v45 )
        goto LABEL_21;
    }
    goto LABEL_23;
  }
LABEL_21:
  if ( (unsigned int)v36 < *(_DWORD *)(a4 + 16) )
  {
    *(_DWORD *)(a4 + 20) = v36 + 1;
    *(_QWORD *)v37 = v44;
    ++*(_QWORD *)a4;
  }
  else
  {
LABEL_50:
    sub_C8CC70(a4, v90[2], v37, v36, v27, v25);
  }
LABEL_23:
  result = *(_BYTE **)(a6 + 16);
  if ( result )
  {
    while ( v44 == *((_QWORD *)result + 3) )
    {
      result = (_BYTE *)*((_QWORD *)result + 1);
      if ( !result )
        return result;
    }
    *a9 = 1;
    return a9;
  }
  return result;
}
