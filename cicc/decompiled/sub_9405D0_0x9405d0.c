// Function: sub_9405D0
// Address: 0x9405d0
//
__int64 __fastcall sub_9405D0(__int64 a1, unsigned int a2, __int64 a3, __int64 a4, __int64 a5, int a6)
{
  _BYTE *v7; // rax
  char v8; // cl
  _BYTE *v9; // rdx
  unsigned __int8 v10; // dl
  _QWORD *v11; // rsi
  const char *v12; // r13
  int v13; // r8d
  __int64 v14; // rsi
  __int64 v15; // rdx
  int v16; // edx
  unsigned __int8 v17; // dl
  _BYTE **v18; // rax
  int v19; // edx
  const char *v20; // rsi
  const char *v21; // r13
  const char **v22; // rax
  int v23; // r9d
  __int64 v24; // rax
  __int64 v25; // rsi
  unsigned int v26; // ecx
  const char **v27; // rdx
  const char *v28; // r8
  __int64 v29; // r12
  unsigned __int8 v31; // dl
  const char **v32; // rax
  int v33; // eax
  _BYTE *v34; // rax
  unsigned __int8 v35; // dl
  _BYTE **v36; // rax
  unsigned __int8 v37; // dl
  __int64 v38; // rax
  const char *v39; // r15
  int v40; // edx
  int v41; // r8d
  int v42; // edx
  int v43; // eax
  __int64 v44; // rax
  unsigned int v45; // esi
  int v46; // r10d
  __int64 v47; // r8
  const char **v48; // rdi
  unsigned int v49; // ecx
  _QWORD *v50; // rax
  const char *v51; // rdx
  __int64 *v52; // r13
  int v53; // eax
  int v54; // edx
  int v55; // edx
  int v56; // eax
  int v57; // ecx
  __int64 v58; // r8
  unsigned int v59; // eax
  const char *v60; // rsi
  int v61; // r10d
  const char **v62; // r9
  int v63; // eax
  int v64; // eax
  __int64 v65; // rsi
  const char **v66; // r8
  unsigned int v67; // r15d
  int v68; // r9d
  const char *v69; // rcx
  int v70; // [rsp+8h] [rbp-78h]
  int v71; // [rsp+8h] [rbp-78h]
  __int128 v72; // [rsp+30h] [rbp-50h]
  __int128 v73; // [rsp+40h] [rbp-40h]

  if ( !a2 )
  {
    v7 = *(_BYTE **)(a1 + 8);
    BYTE8(v73) = 0;
    v8 = *v7;
    v9 = v7;
    if ( *v7 == 16
      || ((v10 = *(v7 - 16), (v10 & 2) != 0)
        ? (v11 = (_QWORD *)*((_QWORD *)v7 - 4))
        : (v11 = &v7[-8 * ((v10 >> 2) & 0xF) - 16]),
          (v9 = (_BYTE *)*v11) != 0) )
    {
      v14 = (unsigned __int8)*(v9 - 16);
      if ( (v14 & 2) != 0 )
      {
        v15 = *((_QWORD *)v9 - 4);
      }
      else
      {
        v14 = 8LL * (((unsigned __int8)v14 >> 2) & 0xF);
        v15 = (__int64)&v9[-v14 - 16];
      }
      v12 = *(const char **)(v15 + 8);
      if ( v12 )
      {
        LODWORD(v12) = sub_B91420(*(_QWORD *)(v15 + 8), v14);
        v7 = *(_BYTE **)(a1 + 8);
        v13 = v16;
        v8 = *v7;
      }
      else
      {
        v13 = 0;
      }
      if ( v8 == 16 )
        goto LABEL_23;
    }
    else
    {
      v12 = byte_3F871B3;
      v13 = 0;
    }
    v17 = *(v7 - 16);
    if ( (v17 & 2) != 0 )
      v18 = (_BYTE **)*((_QWORD *)v7 - 4);
    else
      v18 = (_BYTE **)&v7[-8 * ((v17 >> 2) & 0xF) - 16];
    v7 = *v18;
    if ( !v7 )
    {
      v19 = 0;
      v20 = byte_3F871B3;
      return sub_ADC750((int)a1 + 16, (_DWORD)v20, v19, (_DWORD)v12, v13, a6, v72, v73);
    }
LABEL_23:
    v31 = *(v7 - 16);
    if ( (v31 & 2) != 0 )
      v32 = (const char **)*((_QWORD *)v7 - 4);
    else
      v32 = (const char **)&v7[-8 * ((v31 >> 2) & 0xF) - 16];
    v20 = *v32;
    if ( *v32 )
    {
      v70 = v13;
      v33 = sub_B91420(*v32, v20);
      v13 = v70;
      LODWORD(v20) = v33;
    }
    else
    {
      v19 = 0;
    }
    return sub_ADC750((int)a1 + 16, (_DWORD)v20, v19, (_DWORD)v12, v13, a6, v72, v73);
  }
  v21 = "<unknown>";
  v22 = (const char **)sub_93ED80(a2, 0);
  if ( v22 )
    v21 = *v22;
  v24 = *(unsigned int *)(a1 + 568);
  v25 = *(_QWORD *)(a1 + 552);
  if ( (_DWORD)v24 )
  {
    v26 = (v24 - 1) & (((unsigned int)v21 >> 9) ^ ((unsigned int)v21 >> 4));
    v27 = (const char **)(v25 + 16LL * v26);
    v28 = *v27;
    if ( v21 == *v27 )
    {
LABEL_20:
      if ( v27 != (const char **)(v25 + 16 * v24) )
      {
        v29 = (__int64)v27[1];
        if ( v29 )
          return v29;
      }
    }
    else
    {
      v55 = 1;
      while ( v28 != (const char *)-4096LL )
      {
        v23 = v55 + 1;
        v26 = (v24 - 1) & (v55 + v26);
        v27 = (const char **)(v25 + 16LL * v26);
        v28 = *v27;
        if ( v21 == *v27 )
          goto LABEL_20;
        v55 = v23;
      }
    }
  }
  v34 = *(_BYTE **)(a1 + 8);
  BYTE8(v73) = 0;
  if ( *v34 == 16
    || ((v35 = *(v34 - 16), (v35 & 2) == 0)
      ? (v36 = (_BYTE **)&v34[-8 * ((v35 >> 2) & 0xF) - 16])
      : (v36 = (_BYTE **)*((_QWORD *)v34 - 4)),
        (v34 = *v36) != 0) )
  {
    v37 = *(v34 - 16);
    if ( (v37 & 2) != 0 )
      v38 = *((_QWORD *)v34 - 4);
    else
      v38 = (__int64)&v34[-8 * ((v37 >> 2) & 0xF) - 16];
    v39 = *(const char **)(v38 + 8);
    if ( v39 )
    {
      LODWORD(v39) = sub_B91420(*(_QWORD *)(v38 + 8), v25);
      v41 = v40;
    }
    else
    {
      v41 = 0;
    }
  }
  else
  {
    v41 = 0;
    v39 = byte_3F871B3;
  }
  v42 = 0;
  if ( v21 )
  {
    v71 = v41;
    v43 = strlen(v21);
    v41 = v71;
    v42 = v43;
  }
  v44 = sub_ADC750((int)a1 + 16, (_DWORD)v21, v42, (_DWORD)v39, v41, v23, v72, v73);
  v45 = *(_DWORD *)(a1 + 568);
  v29 = v44;
  if ( !v45 )
  {
    ++*(_QWORD *)(a1 + 544);
    goto LABEL_72;
  }
  v46 = 1;
  v47 = *(_QWORD *)(a1 + 552);
  v48 = 0;
  v49 = (v45 - 1) & (((unsigned int)v21 >> 9) ^ ((unsigned int)v21 >> 4));
  v50 = (_QWORD *)(v47 + 16LL * v49);
  v51 = (const char *)*v50;
  if ( v21 != (const char *)*v50 )
  {
    while ( v51 != (const char *)-4096LL )
    {
      if ( v51 == (const char *)-8192LL && !v48 )
        v48 = (const char **)v50;
      v49 = (v45 - 1) & (v46 + v49);
      v50 = (_QWORD *)(v47 + 16LL * v49);
      v51 = (const char *)*v50;
      if ( v21 == (const char *)*v50 )
        goto LABEL_40;
      ++v46;
    }
    if ( !v48 )
      v48 = (const char **)v50;
    v53 = *(_DWORD *)(a1 + 560);
    ++*(_QWORD *)(a1 + 544);
    v54 = v53 + 1;
    if ( 4 * (v53 + 1) < 3 * v45 )
    {
      if ( v45 - *(_DWORD *)(a1 + 564) - v54 > v45 >> 3 )
      {
LABEL_60:
        *(_DWORD *)(a1 + 560) = v54;
        if ( *v48 != (const char *)-4096LL )
          --*(_DWORD *)(a1 + 564);
        *v48 = v21;
        v52 = (__int64 *)(v48 + 1);
        v48[1] = 0;
        goto LABEL_42;
      }
      sub_9403B0(a1 + 544, v45);
      v63 = *(_DWORD *)(a1 + 568);
      if ( v63 )
      {
        v64 = v63 - 1;
        v65 = *(_QWORD *)(a1 + 552);
        v66 = 0;
        v67 = v64 & (((unsigned int)v21 >> 9) ^ ((unsigned int)v21 >> 4));
        v68 = 1;
        v54 = *(_DWORD *)(a1 + 560) + 1;
        v48 = (const char **)(v65 + 16LL * v67);
        v69 = *v48;
        if ( v21 != *v48 )
        {
          while ( v69 != (const char *)-4096LL )
          {
            if ( v69 == (const char *)-8192LL && !v66 )
              v66 = v48;
            v67 = v64 & (v68 + v67);
            v48 = (const char **)(v65 + 16LL * v67);
            v69 = *v48;
            if ( v21 == *v48 )
              goto LABEL_60;
            ++v68;
          }
          if ( v66 )
            v48 = v66;
        }
        goto LABEL_60;
      }
LABEL_95:
      ++*(_DWORD *)(a1 + 560);
      BUG();
    }
LABEL_72:
    sub_9403B0(a1 + 544, 2 * v45);
    v56 = *(_DWORD *)(a1 + 568);
    if ( v56 )
    {
      v57 = v56 - 1;
      v58 = *(_QWORD *)(a1 + 552);
      v59 = (v56 - 1) & (((unsigned int)v21 >> 9) ^ ((unsigned int)v21 >> 4));
      v54 = *(_DWORD *)(a1 + 560) + 1;
      v48 = (const char **)(v58 + 16LL * v59);
      v60 = *v48;
      if ( v21 != *v48 )
      {
        v61 = 1;
        v62 = 0;
        while ( v60 != (const char *)-4096LL )
        {
          if ( !v62 && v60 == (const char *)-8192LL )
            v62 = v48;
          v59 = v57 & (v61 + v59);
          v48 = (const char **)(v58 + 16LL * v59);
          v60 = *v48;
          if ( v21 == *v48 )
            goto LABEL_60;
          ++v61;
        }
        if ( v62 )
          v48 = v62;
      }
      goto LABEL_60;
    }
    goto LABEL_95;
  }
LABEL_40:
  v52 = v50 + 1;
  if ( v50[1] )
    sub_B91220(v50 + 1);
LABEL_42:
  *v52 = v29;
  if ( v29 )
    sub_B96E90(v52, v29, 1);
  return v29;
}
