// Function: sub_E34DB0
// Address: 0xe34db0
//
_BYTE *__fastcall sub_E34DB0(__int64 a1, unsigned __int8 *a2)
{
  __int64 v2; // rdx
  __int64 v5; // rax
  __int64 v6; // rdx
  __int64 v7; // rbx
  __int64 v8; // rbx
  __int64 v9; // r14
  unsigned int v10; // r15d
  __int64 v11; // rbx
  __int64 v12; // rax
  bool v13; // al
  __int64 v14; // rax
  __int64 v15; // rdx
  __int64 v16; // rbx
  __int64 v17; // rbx
  __int64 v18; // r15
  __int64 v19; // rbx
  __int64 v20; // rax
  unsigned int *v21; // rax
  _BYTE *v22; // r14
  unsigned int v23; // esi
  int v24; // r15d
  __int64 v25; // rdi
  unsigned int v26; // ecx
  _QWORD *v27; // rdx
  unsigned __int8 **v28; // rax
  unsigned __int8 *v29; // r11
  _QWORD *v30; // rax
  const char *v31; // rax
  int v33; // ecx
  int v34; // ecx
  int v35; // r9d
  int v36; // r9d
  __int64 v37; // r10
  unsigned int v38; // edx
  unsigned __int8 *v39; // r8
  int v40; // edi
  unsigned __int8 **v41; // rsi
  int v42; // r8d
  int v43; // r8d
  __int64 v44; // r9
  unsigned __int8 **v45; // rdi
  __int64 v46; // rbx
  int v47; // edx
  unsigned __int8 *v48; // rsi
  unsigned __int8 *v49; // [rsp+0h] [rbp-B0h]
  __int64 v50; // [rsp+8h] [rbp-A8h]
  _QWORD v51[2]; // [rsp+10h] [rbp-A0h] BYREF
  void (__fastcall *v52)(_QWORD *, _QWORD *, __int64); // [rsp+20h] [rbp-90h]
  char v53; // [rsp+30h] [rbp-80h]
  char v54; // [rsp+31h] [rbp-7Fh]
  _QWORD v55[2]; // [rsp+40h] [rbp-70h] BYREF
  void (__fastcall *v56)(_QWORD *, _QWORD *, __int64); // [rsp+50h] [rbp-60h]
  char v57; // [rsp+60h] [rbp-50h] BYREF
  char v58; // [rsp+61h] [rbp-4Fh]
  void (__fastcall *v59)(char *, char *, __int64); // [rsp+70h] [rbp-40h]

  if ( (unsigned __int8)(*a2 - 34) > 0x33u )
    return 0;
  v2 = 0x8000000000041LL;
  if ( !_bittest64(&v2, (unsigned int)*a2 - 34) || (a2[7] & 0x80u) == 0 )
    return 0;
  v5 = sub_BD2BC0((__int64)a2);
  v7 = v5 + v6;
  if ( (a2[7] & 0x80u) != 0 )
    v7 -= sub_BD2BC0((__int64)a2);
  v8 = v7 >> 4;
  if ( !(_DWORD)v8 )
    return 0;
  v9 = 0;
  v10 = 0;
  v11 = 16LL * (unsigned int)v8;
  do
  {
    v12 = 0;
    if ( (a2[7] & 0x80u) != 0 )
      v12 = sub_BD2BC0((__int64)a2);
    v13 = *(_DWORD *)(*(_QWORD *)(v12 + v9) + 8LL) == 9;
    v9 += 16;
    v10 += v13;
  }
  while ( v11 != v9 );
  if ( v10 > 1 )
  {
    sub_E45390(v51, a1 + 144, a2);
    v58 = 1;
    v31 = "The 'convergencectrl' bundle can occur at most once on a call";
LABEL_32:
    v55[0] = v31;
    v57 = 3;
    sub_E348A0((_BYTE *)a1, (__int64)v55, v51, 1);
    if ( v52 )
      v52(v51, v51, 3);
    return 0;
  }
  if ( !v10 )
    return 0;
  if ( (a2[7] & 0x80u) != 0 )
  {
    v14 = sub_BD2BC0((__int64)a2);
    v16 = v14 + v15;
    if ( (a2[7] & 0x80u) != 0 )
      v16 -= sub_BD2BC0((__int64)a2);
    v17 = v16 >> 4;
    if ( (_DWORD)v17 )
    {
      v18 = 0;
      v19 = 16LL * (unsigned int)v17;
      while ( 1 )
      {
        v20 = 0;
        if ( (a2[7] & 0x80u) != 0 )
          v20 = sub_BD2BC0((__int64)a2);
        v21 = (unsigned int *)(v18 + v20);
        if ( *(_DWORD *)(*(_QWORD *)v21 + 8LL) == 9 )
          break;
        v18 += 16;
        if ( v18 == v19 )
          goto LABEL_23;
      }
      v49 = &a2[32LL * v21[2] - 32LL * (*((_DWORD *)a2 + 1) & 0x7FFFFFF)];
      v50 = (32LL * v21[3] - 32LL * v21[2]) >> 5;
    }
  }
LABEL_23:
  if ( v50 != 1 || (v22 = *(_BYTE **)v49, *(_BYTE *)(*(_QWORD *)(*(_QWORD *)v49 + 8LL) + 8LL) != 11) )
  {
    sub_E45390(v51, a1 + 144, a2);
    v58 = 1;
    v31 = "The 'convergencectrl' bundle requires exactly one token use.";
    goto LABEL_32;
  }
  if ( *v22 <= 0x1Cu || (unsigned int)sub_E345B0(*(_BYTE **)v49) == 3 )
  {
    sub_E45370(v55, a1 + 144, v22);
    sub_E45390(&v57, a1 + 144, a2);
    v54 = 1;
    v51[0] = "Convergence control tokens can only be produced by calls to the convergence control intrinsics.";
    v53 = 3;
    sub_E348A0((_BYTE *)a1, (__int64)v51, v55, 2);
    if ( v59 )
      v59(&v57, &v57, 3);
    if ( v56 )
      v56(v55, v55, 3);
    return 0;
  }
  v23 = *(_DWORD *)(a1 + 184);
  if ( !v23 )
  {
    ++*(_QWORD *)(a1 + 160);
    goto LABEL_55;
  }
  v24 = 1;
  v25 = *(_QWORD *)(a1 + 168);
  v26 = (v23 - 1) & (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4));
  v27 = (_QWORD *)(v25 + 16LL * v26);
  v28 = 0;
  v29 = (unsigned __int8 *)*v27;
  if ( a2 != (unsigned __int8 *)*v27 )
  {
    while ( v29 != (unsigned __int8 *)-4096LL )
    {
      if ( v29 == (unsigned __int8 *)-8192LL && !v28 )
        v28 = (unsigned __int8 **)v27;
      v26 = (v23 - 1) & (v24 + v26);
      v27 = (_QWORD *)(v25 + 16LL * v26);
      v29 = (unsigned __int8 *)*v27;
      if ( a2 == (unsigned __int8 *)*v27 )
        goto LABEL_29;
      ++v24;
    }
    v33 = *(_DWORD *)(a1 + 176);
    if ( !v28 )
      v28 = (unsigned __int8 **)v27;
    ++*(_QWORD *)(a1 + 160);
    v34 = v33 + 1;
    if ( 4 * v34 < 3 * v23 )
    {
      if ( v23 - *(_DWORD *)(a1 + 180) - v34 > v23 >> 3 )
      {
LABEL_51:
        *(_DWORD *)(a1 + 176) = v34;
        if ( *v28 != (unsigned __int8 *)-4096LL )
          --*(_DWORD *)(a1 + 180);
        *v28 = a2;
        v30 = v28 + 1;
        *v30 = 0;
        goto LABEL_30;
      }
      sub_E34BD0(a1 + 160, v23);
      v42 = *(_DWORD *)(a1 + 184);
      if ( v42 )
      {
        v43 = v42 - 1;
        v44 = *(_QWORD *)(a1 + 168);
        v45 = 0;
        LODWORD(v46) = v43 & (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4));
        v47 = 1;
        v34 = *(_DWORD *)(a1 + 176) + 1;
        v28 = (unsigned __int8 **)(v44 + 16LL * (unsigned int)v46);
        v48 = *v28;
        if ( a2 != *v28 )
        {
          while ( v48 != (unsigned __int8 *)-4096LL )
          {
            if ( !v45 && v48 == (unsigned __int8 *)-8192LL )
              v45 = v28;
            v46 = v43 & (unsigned int)(v46 + v47);
            v28 = (unsigned __int8 **)(v44 + 16 * v46);
            v48 = *v28;
            if ( a2 == *v28 )
              goto LABEL_51;
            ++v47;
          }
          if ( v45 )
            v28 = v45;
        }
        goto LABEL_51;
      }
LABEL_78:
      ++*(_DWORD *)(a1 + 176);
      BUG();
    }
LABEL_55:
    sub_E34BD0(a1 + 160, 2 * v23);
    v35 = *(_DWORD *)(a1 + 184);
    if ( v35 )
    {
      v36 = v35 - 1;
      v37 = *(_QWORD *)(a1 + 168);
      v38 = v36 & (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4));
      v34 = *(_DWORD *)(a1 + 176) + 1;
      v28 = (unsigned __int8 **)(v37 + 16LL * v38);
      v39 = *v28;
      if ( a2 != *v28 )
      {
        v40 = 1;
        v41 = 0;
        while ( v39 != (unsigned __int8 *)-4096LL )
        {
          if ( !v41 && v39 == (unsigned __int8 *)-8192LL )
            v41 = v28;
          v38 = v36 & (v40 + v38);
          v28 = (unsigned __int8 **)(v37 + 16LL * v38);
          v39 = *v28;
          if ( a2 == *v28 )
            goto LABEL_51;
          ++v40;
        }
        if ( v41 )
          v28 = v41;
      }
      goto LABEL_51;
    }
    goto LABEL_78;
  }
LABEL_29:
  v30 = v27 + 1;
LABEL_30:
  *v30 = v22;
  return v22;
}
