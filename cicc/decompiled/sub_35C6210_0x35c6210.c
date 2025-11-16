// Function: sub_35C6210
// Address: 0x35c6210
//
__int64 __fastcall sub_35C6210(__int64 a1, __int64 *a2, __int64 a3, char a4, unsigned int a5, char a6)
{
  __int64 (__fastcall *v7)(_QWORD); // rax
  __int64 v8; // rdx
  __int64 v9; // rbx
  __int64 v10; // rax
  unsigned __int64 v11; // rax
  __int64 v12; // rax
  __int64 v13; // r8
  __int64 v14; // r9
  _BYTE *v15; // rdi
  int v16; // ecx
  unsigned int v17; // eax
  unsigned int v18; // r12d
  __int64 v19; // r13
  int v20; // ecx
  char v21; // r13
  __int64 *v22; // r9
  unsigned __int64 v23; // r12
  unsigned __int16 *v24; // r15
  _BYTE *v25; // rdi
  unsigned int v26; // ecx
  __int16 *v27; // rsi
  int v28; // edx
  unsigned __int16 v29; // dx
  __int64 v30; // rax
  __int64 v31; // rcx
  _QWORD *v32; // rax
  _QWORD *v33; // rdx
  __int64 v34; // rax
  __int64 i; // r12
  unsigned __int16 *v36; // rax
  __int64 v37; // r11
  unsigned int v38; // ecx
  __int16 *v39; // r8
  int v40; // r11d
  unsigned __int16 v41; // r13
  unsigned __int16 *v42; // rsi
  __int64 v43; // r8
  unsigned int v44; // ecx
  __int16 *v45; // rdx
  int v46; // r11d
  __int64 v47; // rdx
  unsigned int v48; // ecx
  __int16 *v49; // r8
  int v50; // edx
  __int64 result; // rax
  __int64 v52; // rax
  __int64 v53; // rcx
  unsigned __int64 v54; // rax
  __int64 j; // rax
  __int64 v56; // rdx
  __int64 v57; // rcx
  __int64 v58; // rax
  unsigned int v59; // ecx
  __int64 v60; // rsi
  unsigned __int64 v61; // rax
  __int64 v63; // [rsp+10h] [rbp-E0h]
  _QWORD *v67; // [rsp+30h] [rbp-C0h]
  _QWORD *v68; // [rsp+38h] [rbp-B8h]
  __int64 v69; // [rsp+40h] [rbp-B0h]
  __int64 v70; // [rsp+48h] [rbp-A8h]
  unsigned __int64 v71; // [rsp+50h] [rbp-A0h]
  int v73; // [rsp+60h] [rbp-90h]
  unsigned __int16 v75; // [rsp+66h] [rbp-8Ah]
  __int64 *v76; // [rsp+68h] [rbp-88h]
  __int64 *v77; // [rsp+68h] [rbp-88h]
  __int64 v78; // [rsp+70h] [rbp-80h] BYREF
  _BYTE *v79; // [rsp+78h] [rbp-78h] BYREF
  __int64 v80; // [rsp+80h] [rbp-70h]
  _BYTE s[48]; // [rsp+88h] [rbp-68h] BYREF
  int v82; // [rsp+B8h] [rbp-38h]

  v7 = (__int64 (__fastcall *)(_QWORD))a2[8];
  v63 = *(_QWORD *)(a3 + 24);
  if ( v7 )
  {
    v70 = v7(*(_QWORD *)(*(_QWORD *)(a3 + 24) + 32LL));
    v9 = v8;
  }
  else
  {
    v9 = *(unsigned __int16 *)(*a2 + 20);
    v70 = *(_QWORD *)*a2;
  }
  if ( (**(_QWORD **)(a1 + 32) & 0xFFFFFFFFFFFFFFF8LL) == 0 )
    BUG();
  v71 = **(_QWORD **)(a1 + 32) & 0xFFFFFFFFFFFFFFF8LL;
  v10 = *(_QWORD *)v71;
  if ( (*(_QWORD *)v71 & 4) == 0 && (*(_BYTE *)(v71 + 44) & 4) != 0 )
  {
    while ( 1 )
    {
      v11 = v10 & 0xFFFFFFFFFFFFFFF8LL;
      if ( (*(_BYTE *)(v11 + 44) & 4) == 0 )
        break;
      v10 = *(_QWORD *)v11;
    }
    v71 = v11;
  }
  v68 = *(_QWORD **)(a1 + 16);
  v69 = *(_QWORD *)(v71 + 24);
  v12 = (*(__int64 (__fastcall **)(_QWORD))(**(_QWORD **)(*v68 + 16LL) + 200LL))(*(_QWORD *)(*v68 + 16LL));
  v15 = s;
  v82 = 0;
  v78 = v12;
  v79 = s;
  v80 = 0x600000000LL;
  v16 = *(_DWORD *)(v12 + 44);
  v82 = v16;
  v17 = (unsigned int)(v16 + 63) >> 6;
  v18 = v17;
  if ( v17 )
  {
    v19 = v17;
    if ( v17 > 6 )
    {
      sub_C8D5F0((__int64)&v79, s, v17, 8u, v13, v14);
      v15 = &v79[8 * (unsigned int)v80];
    }
    memset(v15, 0, 8 * v19);
    LODWORD(v80) = v18 + v80;
    LOBYTE(v16) = v82;
  }
  v20 = v16 & 0x3F;
  if ( v20 )
    *(_QWORD *)&v79[8 * (unsigned int)v80 - 8] &= ~(-1LL << v20);
  v21 = 0;
  v22 = &v78;
  v67 = 0;
  v73 = 25;
  v23 = v71;
  v24 = (unsigned __int16 *)(v70 + 2 * v9);
  v75 = 0;
  while ( 1 )
  {
    v76 = v22;
    sub_2E22100(v22, v23);
    v22 = v76;
    if ( a3 != v23 )
    {
      if ( !v21 )
        goto LABEL_34;
      goto LABEL_19;
    }
    if ( v24 != (unsigned __int16 *)v70 )
      break;
LABEL_83:
    if ( a4 )
    {
      v61 = v71;
      if ( (*(_BYTE *)v71 & 4) == 0 && (*(_BYTE *)(v71 + 44) & 8) != 0 )
      {
        do
          v61 = *(_QWORD *)(v61 + 8);
        while ( (*(_BYTE *)(v61 + 44) & 8) != 0 );
      }
      sub_2E22100(v76, *(_QWORD *)(v61 + 8));
      v67 = (_QWORD *)v23;
      v22 = v76;
    }
    else
    {
      v67 = (_QWORD *)v23;
    }
LABEL_19:
    v25 = v79;
    if ( (*(_BYTE *)(v71 + 44) & 1) == 0 && (*(_BYTE *)(v23 + 44) & 1) != 0 )
      goto LABEL_50;
    if ( v75 )
    {
      v26 = *(_DWORD *)(*(_QWORD *)(v78 + 8) + 24LL * v75 + 16) & 0xFFF;
      v27 = (__int16 *)(*(_QWORD *)(v78 + 56) + 2LL * (*(_DWORD *)(*(_QWORD *)(v78 + 8) + 24LL * v75 + 16) >> 12));
      do
      {
        if ( !v27 )
          break;
        if ( (*(_QWORD *)&v79[8 * (v26 >> 6)] & (1LL << v26)) != 0 )
          goto LABEL_41;
        v28 = *v27++;
        v26 += v28;
      }
      while ( (_WORD)v28 );
      v29 = v75;
    }
    else
    {
LABEL_41:
      v36 = (unsigned __int16 *)v70;
      if ( v24 == (unsigned __int16 *)v70 )
        goto LABEL_50;
      while ( 1 )
      {
        v29 = *v36;
        if ( (*(_QWORD *)(v68[48] + 8 * ((unsigned __int64)*v36 >> 6)) & (1LL << *v36)) == 0 )
          break;
LABEL_43:
        if ( v24 == ++v36 )
          goto LABEL_50;
      }
      v37 = *(_QWORD *)(v78 + 8);
      v38 = *(_DWORD *)(v37 + 24LL * v29 + 16) & 0xFFF;
      v39 = (__int16 *)(*(_QWORD *)(v78 + 56) + 2LL * (*(_DWORD *)(v37 + 24LL * v29 + 16) >> 12));
      do
      {
        if ( !v39 )
          break;
        if ( (*(_QWORD *)&v79[8 * (v38 >> 6)] & (1LL << v38)) != 0 )
          goto LABEL_43;
        v40 = *v39++;
        v38 += v40;
      }
      while ( (_WORD)v40 );
      if ( !v29 )
      {
LABEL_50:
        v41 = v75;
        goto LABEL_65;
      }
    }
    if ( !--v73 )
    {
      v41 = v29;
      goto LABEL_65;
    }
    v30 = *(_QWORD *)(v23 + 32);
    v31 = v30 + 40LL * (*(_DWORD *)(v23 + 40) & 0xFFFFFF);
    if ( v30 != v31 )
    {
      while ( *(_BYTE *)v30 || *(int *)(v30 + 8) >= 0 )
      {
        v30 += 40;
        if ( v31 == v30 )
          goto LABEL_32;
      }
      v67 = (_QWORD *)v23;
      v73 = 25;
    }
LABEL_32:
    v75 = v29;
    if ( *(_QWORD *)(v69 + 56) == v23 )
      goto LABEL_50;
    v21 = 1;
LABEL_34:
    v32 = (_QWORD *)(*(_QWORD *)v23 & 0xFFFFFFFFFFFFFFF8LL);
    v33 = v32;
    if ( !v32 )
      BUG();
    v23 = *(_QWORD *)v23 & 0xFFFFFFFFFFFFFFF8LL;
    v34 = *v32;
    if ( (v34 & 4) == 0 && (*((_BYTE *)v33 + 44) & 4) != 0 )
    {
      for ( i = v34; ; i = *(_QWORD *)v23 )
      {
        v23 = i & 0xFFFFFFFFFFFFFFF8LL;
        if ( (*(_BYTE *)(v23 + 44) & 4) == 0 )
          break;
      }
    }
  }
  v42 = (unsigned __int16 *)v70;
  while ( 1 )
  {
    v41 = *v42;
    if ( (*(_QWORD *)(v68[48] + 8 * ((unsigned __int64)*v42 >> 6)) & (1LL << *v42)) == 0 )
      break;
LABEL_54:
    if ( v24 == ++v42 )
      goto LABEL_83;
  }
  v25 = v79;
  v43 = 24LL * v41;
  v44 = *(_DWORD *)(*(_QWORD *)(v78 + 8) + v43 + 16) & 0xFFF;
  v45 = (__int16 *)(*(_QWORD *)(v78 + 56) + 2LL * (*(_DWORD *)(*(_QWORD *)(v78 + 8) + v43 + 16) >> 12));
  do
  {
    if ( !v45 )
      break;
    if ( (*(_QWORD *)&v79[8 * (v44 >> 6)] & (1LL << v44)) != 0 )
      goto LABEL_54;
    v46 = *v45++;
    v44 += v46;
  }
  while ( (_WORD)v46 );
  v47 = *(_QWORD *)(a1 + 88);
  v48 = *(_DWORD *)(*(_QWORD *)(v47 + 8) + v43 + 16) & 0xFFF;
  v49 = (__int16 *)(*(_QWORD *)(v47 + 56) + 2LL * (*(_DWORD *)(*(_QWORD *)(v47 + 8) + v43 + 16) >> 12));
  do
  {
    if ( !v49 )
      break;
    if ( (*(_QWORD *)(*(_QWORD *)(a1 + 96) + 8LL * (v48 >> 6)) & (1LL << v48)) != 0 )
      goto LABEL_54;
    v50 = *v49++;
    v48 += v50;
  }
  while ( (_WORD)v50 );
  v67 = (_QWORD *)(v69 + 48);
LABEL_65:
  if ( v25 != s )
  {
    v77 = v22;
    _libc_free((unsigned __int64)v25);
    v22 = v77;
  }
  if ( v41 && v67 == (_QWORD *)(v63 + 48) )
    return v41;
  result = 0;
  if ( a6 )
  {
    v52 = *(_QWORD *)(a1 + 32);
    if ( a4 )
    {
      if ( !v52 )
        BUG();
      if ( (*(_BYTE *)v52 & 4) == 0 && (*(_BYTE *)(v52 + 44) & 8) != 0 )
      {
        do
          v52 = *(_QWORD *)(v52 + 8);
        while ( (*(_BYTE *)(v52 + 44) & 8) != 0 );
      }
      v78 = *(_QWORD *)(v52 + 8);
    }
    else
    {
      v78 = *(_QWORD *)(a1 + 32);
    }
    v53 = sub_35C5CB0((unsigned int *)a1, v41, a2, a5, v67, v22);
    v54 = *v67 & 0xFFFFFFFFFFFFFFF8LL;
    if ( !v54 )
      BUG();
    if ( (*(_QWORD *)v54 & 4) == 0 && (*(_BYTE *)(v54 + 44) & 4) != 0 )
    {
      for ( j = *(_QWORD *)v54; ; j = *(_QWORD *)v54 )
      {
        v54 = j & 0xFFFFFFFFFFFFFFF8LL;
        if ( (*(_BYTE *)(v54 + 44) & 4) == 0 )
          break;
      }
    }
    *(_QWORD *)(v53 + 8) = v54;
    v56 = *(_QWORD *)(a1 + 88);
    v57 = *(_QWORD *)(v56 + 8);
    v58 = *(_DWORD *)(v57 + 24LL * v41 + 16) >> 12;
    v59 = *(_DWORD *)(v57 + 24LL * v41 + 16) & 0xFFF;
    v60 = *(_QWORD *)(v56 + 56) + 2 * v58;
    do
    {
      if ( !v60 )
        break;
      v60 += 2;
      *(_QWORD *)(*(_QWORD *)(a1 + 96) + 8LL * (v59 >> 6)) &= ~(1LL << v59);
      v59 += *(__int16 *)(v60 - 2);
    }
    while ( *(_WORD *)(v60 - 2) );
    return v41;
  }
  return result;
}
