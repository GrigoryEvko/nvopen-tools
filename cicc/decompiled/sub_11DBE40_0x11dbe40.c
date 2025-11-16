// Function: sub_11DBE40
// Address: 0x11dbe40
//
__int64 __fastcall sub_11DBE40(__int64 a1, char **a2, __int64 a3, unsigned __int64 a4, char a5, __int64 a6)
{
  unsigned __int64 v6; // r13
  unsigned __int64 v8; // rcx
  char *v9; // rdx
  unsigned __int64 i; // rbx
  char v11; // al
  char *v12; // rsi
  char *v13; // rcx
  __int64 v15; // r14
  __int64 v16; // rax
  __int64 v17; // rdx
  __int64 v18; // rax
  char v19; // r8
  char v20; // dl
  unsigned __int64 v21; // r9
  unsigned __int64 v22; // r8
  char *v23; // r10
  char v24; // dl
  int v25; // edi
  unsigned __int64 v26; // r12
  unsigned __int64 v27; // rax
  unsigned __int64 v28; // rsi
  unsigned __int64 v29; // rcx
  unsigned int v30; // edx
  __int64 v31; // rdx
  unsigned __int64 v32; // rdx
  unsigned __int64 v33; // rcx
  __int64 v34; // rax
  __int64 v35; // rdx
  char v36; // al
  __int64 v37; // r13
  __int64 v38; // rax
  int v39; // eax
  __int64 v40; // r13
  __int64 v41; // rax
  __int64 v42; // r13
  __int64 v43; // rax
  char v44; // al
  _QWORD *v45; // rax
  __int64 v46; // r9
  __int64 v47; // rbx
  unsigned int *v48; // r15
  __int64 v49; // r13
  __int64 v50; // rdx
  unsigned int v51; // esi
  char v56; // [rsp+10h] [rbp-80h]
  char v57; // [rsp+17h] [rbp-79h]
  _BYTE *v59; // [rsp+28h] [rbp-68h] BYREF
  const char *v60; // [rsp+30h] [rbp-60h] BYREF
  __int64 v61; // [rsp+38h] [rbp-58h]
  __int16 v62; // [rsp+50h] [rbp-40h]

  v6 = a4;
  if ( a4 - 2 > 0x22 && a4 )
    return 0;
  v8 = (unsigned __int64)a2[1];
  if ( !v8 )
    return 0;
  v9 = *a2;
  for ( i = 0; i != v8; ++i )
  {
    v11 = v9[i];
    v12 = &v9[i];
    if ( v11 != 32 && (unsigned __int8)(v11 - 9) > 4u )
    {
      if ( v8 < i )
      {
        a2[1] = 0;
        *a2 = &v9[v8];
        return 0;
      }
      v8 -= i;
      *a2 = v12;
      a2[1] = (char *)v8;
      if ( !v8 )
        return 0;
      v9 += i;
      v57 = *v12;
      if ( ((*v12 - 43) & 0xFD) == 0 )
        goto LABEL_10;
LABEL_14:
      v15 = *(_QWORD *)(a1 + 8);
      v16 = sub_BCAE30(v15);
      v61 = v17;
      v60 = (const char *)v16;
      LODWORD(v18) = sub_CA1930(&v60);
      v19 = a5;
      v20 = v18;
      goto LABEL_15;
    }
  }
  v57 = *v9;
  if ( ((*v9 - 43) & 0xFD) != 0 )
    goto LABEL_14;
LABEL_10:
  v13 = (char *)(v8 - 1);
  *a2 = v9 + 1;
  a2[1] = v13;
  if ( !v13 )
    return 0;
  ++i;
  v15 = *(_QWORD *)(a1 + 8);
  v34 = sub_BCAE30(v15);
  v61 = v35;
  v60 = (const char *)v34;
  LODWORD(v18) = sub_CA1930(&v60);
  v19 = a5;
  v20 = v18;
  if ( v57 == 45 && a5 )
  {
    v21 = 1;
    v18 = (unsigned int)v18;
    goto LABEL_46;
  }
LABEL_15:
  v21 = (unsigned int)v18;
  v18 = (unsigned int)v18;
  if ( v19 )
  {
    v21 = 0;
LABEL_46:
    if ( v18 )
    {
      v22 = (unsigned __int64)a2[1];
      v21 = v21 + (1LL << (v20 - 1)) - 1;
      if ( v22 > 1 )
      {
LABEL_19:
        v23 = *a2;
        v24 = **a2;
        if ( v24 == 48 )
        {
          v36 = v23[1];
          if ( (unsigned __int8)(v36 - 97) < 0x1Au )
            v36 -= 32;
          if ( v36 == 88 )
          {
            if ( v22 == 2 || (v6 & 0xFFFFFFFFFFFFFFEFLL) != 0 )
              return 0;
            v22 -= 2LL;
            i += 2LL;
            v6 = 16;
            *a2 = v23 + 2;
            a2[1] = (char *)v22;
            v24 = v23[2];
            v23 += 2;
          }
          else if ( !v6 )
          {
            v6 = 8;
          }
        }
        else if ( !v6 )
        {
          v6 = 10;
        }
        goto LABEL_22;
      }
      goto LABEL_48;
    }
  }
  else if ( (_DWORD)v18 )
  {
    v21 = 0xFFFFFFFFFFFFFFFFLL >> (64 - v20);
  }
  v22 = (unsigned __int64)a2[1];
  if ( v22 > 1 )
    goto LABEL_19;
LABEL_48:
  if ( !v6 )
    v6 = 10;
  if ( !v22 )
  {
    v26 = 0;
LABEL_62:
    if ( a3 )
    {
      v37 = i + v22;
      v38 = sub_BCB2E0(*(_QWORD **)(a6 + 72));
      v59 = (_BYTE *)sub_ACD640(v38, v37, 0);
      v39 = *(_DWORD *)(a1 + 4);
      v62 = 259;
      v40 = *(_QWORD *)(a1 - 32LL * (v39 & 0x7FFFFFF));
      v60 = "endptr";
      v41 = sub_BCB2B0(*(_QWORD **)(a6 + 72));
      v42 = sub_921130((unsigned int **)a6, v41, v40, &v59, 1, (__int64)&v60, 3u);
      v43 = sub_AA4E30(*(_QWORD *)(a6 + 48));
      v44 = sub_AE5020(v43, *(_QWORD *)(v42 + 8));
      v62 = 257;
      v56 = v44;
      v45 = sub_BD2C40(80, unk_3F10A10);
      v47 = (__int64)v45;
      if ( v45 )
        sub_B4D3C0((__int64)v45, v42, a3, 0, v56, v46, 0, 0);
      (*(void (__fastcall **)(_QWORD, __int64, const char **, _QWORD, _QWORD))(**(_QWORD **)(a6 + 88) + 16LL))(
        *(_QWORD *)(a6 + 88),
        v47,
        &v60,
        *(_QWORD *)(a6 + 56),
        *(_QWORD *)(a6 + 64));
      v48 = *(unsigned int **)a6;
      v49 = *(_QWORD *)a6 + 16LL * *(unsigned int *)(a6 + 8);
      if ( *(_QWORD *)a6 != v49 )
      {
        do
        {
          v50 = *((_QWORD *)v48 + 1);
          v51 = *v48;
          v48 += 4;
          sub_B99FD0(v47, v51, v50);
        }
        while ( (unsigned int *)v49 != v48 );
      }
    }
    if ( v57 == 45 )
      v26 = -(__int64)v26;
    return sub_AD64C0(v15, v26, 0);
  }
  v23 = *a2;
  v22 = 1;
  v24 = **a2;
LABEL_22:
  v25 = 0;
  v26 = 0;
  while ( 1 )
  {
    LOBYTE(v27) = v24 - 48;
    if ( (unsigned __int8)(v24 - 48) > 9u )
    {
      if ( (unsigned __int8)(v24 - 97) <= 0x19u )
      {
        LOBYTE(v27) = v24 - 87;
      }
      else
      {
        if ( (unsigned __int8)(v24 - 65) > 0x19u )
          return 0;
        LOBYTE(v27) = v24 - 55;
      }
    }
    v27 = (unsigned __int8)v27;
    if ( (unsigned __int8)v27 >= v6 )
      return 0;
    if ( v26
      && (_BitScanReverse64(&v28, v6),
          _BitScanReverse64(&v29, v26),
          v30 = 63 - (v29 ^ 0x3F) + 63 - (v28 ^ 0x3F),
          v30 > 0x3E) )
    {
      if ( v30 != 63 )
        return 0;
      v31 = v6 * (v26 >> 1);
      if ( v31 < 0 )
        return 0;
      v32 = 2 * v31;
      if ( (v26 & 1) != 0 )
      {
        v33 = v32 + v6;
        if ( v32 < v6 )
          v32 = v6;
        if ( v33 < v32 )
          return 0;
        v32 = v33;
      }
    }
    else
    {
      v32 = v6 * v26;
    }
    v26 = (unsigned __int8)v27 + v32;
    if ( (unsigned __int8)v27 < v32 )
      v27 = v32;
    if ( v26 < v27 || v26 > v21 )
      return 0;
    if ( ++v25 == v22 )
      goto LABEL_62;
    v24 = v23[v25];
  }
}
