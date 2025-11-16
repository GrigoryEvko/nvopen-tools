// Function: sub_6DFA90
// Address: 0x6dfa90
//
__int64 __fastcall sub_6DFA90(__int64 a1, _DWORD *a2, _DWORD *a3)
{
  unsigned int v3; // r15d
  bool v4; // zf
  __int64 v6; // r14
  char v9; // al
  __int64 *v10; // r12
  __int64 *v11; // rbx
  char v12; // dl
  __int64 v13; // rbx
  __int64 v14; // rdx
  __int64 v15; // rcx
  __int64 v16; // r8
  char v17; // al
  __int64 *v18; // rdi
  __int64 v19; // r12
  _DWORD *v20; // r9
  _DWORD *v21; // r9
  __int64 k; // r8
  __int64 v23; // rdi
  __int64 v24; // rax
  __int64 v25; // rdx
  __int64 v26; // rcx
  __int64 v27; // r8
  _DWORD *v28; // r9
  __int64 m; // rsi
  int v30; // eax
  unsigned __int64 v31; // rsi
  __int64 v32; // r15
  int v33; // eax
  __int64 v34; // rax
  __int64 i; // rax
  __int64 v36; // rax
  __int64 v37; // rcx
  __int64 v38; // r8
  __int64 v39; // rdi
  int v40; // eax
  _DWORD *v41; // rdi
  _DWORD *v42; // rsi
  __int64 n; // rcx
  int v44; // eax
  __int64 v45; // rax
  _DWORD *v46; // [rsp+0h] [rbp-70h]
  __int64 v47; // [rsp+8h] [rbp-68h]
  _DWORD *v48; // [rsp+8h] [rbp-68h]
  _DWORD *v49; // [rsp+10h] [rbp-60h]
  unsigned __int64 v50; // [rsp+10h] [rbp-60h]
  __int64 v51; // [rsp+20h] [rbp-50h]
  __int64 v52; // [rsp+20h] [rbp-50h]
  _DWORD *v53; // [rsp+20h] [rbp-50h]
  _DWORD *v54; // [rsp+28h] [rbp-48h]
  _DWORD *v55; // [rsp+28h] [rbp-48h]
  __int64 j; // [rsp+28h] [rbp-48h]
  __int64 v57; // [rsp+28h] [rbp-48h]
  int v58; // [rsp+30h] [rbp-40h] BYREF
  _BOOL4 v59; // [rsp+34h] [rbp-3Ch] BYREF
  _QWORD v60[7]; // [rsp+38h] [rbp-38h] BYREF

  v3 = 1;
  *a2 = 0;
  *a3 = 0;
  v4 = *(_BYTE *)(a1 + 24) == 1;
  v58 = 0;
  if ( !v4 )
    return v3;
  v6 = a1;
  if ( *(_BYTE *)(a1 + 56) == 3 )
  {
    v6 = *(_QWORD *)(a1 + 72);
    if ( *(_BYTE *)(v6 + 24) != 1 )
      return 1;
    v12 = *(_BYTE *)(v6 + 56);
    if ( v12 != 21 )
    {
      if ( (unsigned __int8)(v12 - 50) > 1u || (*(_BYTE *)(qword_4D03C50 + 17LL) & 1) == 0 )
        return 1;
      goto LABEL_7;
    }
    if ( (*(_BYTE *)(qword_4D03C50 + 17LL) & 1) == 0 )
      return 1;
    v6 = *(_QWORD *)(v6 + 72);
    if ( *(_BYTE *)(v6 + 24) != 1 )
      return 1;
  }
  else if ( (*(_BYTE *)(qword_4D03C50 + 17LL) & 1) == 0 )
  {
    return 1;
  }
  v9 = *(_BYTE *)(v6 + 56);
  if ( (unsigned __int8)(v9 - 50) > 1u && v9 != 92 )
    return 1;
LABEL_7:
  v10 = *(__int64 **)(v6 + 72);
  v54 = a3;
  v11 = (__int64 *)v10[2];
  if ( (unsigned int)sub_8D2E30(*v11) )
  {
    v11 = v10;
    v10 = (__int64 *)v10[2];
  }
  if ( *((_BYTE *)v11 + 24) != 2 )
    return 1;
  v13 = v11[7];
  if ( *(_BYTE *)(v13 + 173) != 1 )
    return 1;
  v51 = *v10;
  if ( !(unsigned int)sub_8D2E30(*v10) )
    return 1;
  v17 = *((_BYTE *)v10 + 24);
  if ( v17 != 2 )
  {
    if ( v17 == 1 && *((_BYTE *)v10 + 56) == 21 )
    {
      v18 = (__int64 *)v10[9];
      v19 = *v18;
      sub_6DFA90(v18, &v58, v60);
      v20 = v54;
      goto LABEL_22;
    }
    return 1;
  }
  v32 = v10[7];
  v33 = sub_717530(v32, v60, v14, v15, v16, v54);
  v20 = v54;
  if ( !v33 )
  {
    if ( *(_BYTE *)(v32 + 173) == 6
      && *(_BYTE *)(v32 + 176) == 1
      && (*(_BYTE *)(v32 + 168) & 8) != 0
      && (unsigned int)sub_8D2E30(*(_QWORD *)(v32 + 128)) )
    {
      v19 = *(_QWORD *)(*(_QWORD *)(v32 + 184) + 120LL);
      v34 = sub_8D46C0(*(_QWORD *)(v32 + 128));
      v20 = v54;
      v57 = v34;
      for ( i = v19; *(_BYTE *)(i + 140) == 12; i = *(_QWORD *)(i + 160) )
        ;
      v50 = *(_QWORD *)(i + 128);
      while ( 1 )
      {
        v48 = v20;
        if ( !(unsigned int)sub_8D3410(v19) )
          break;
        v36 = sub_8D4050(v19);
        v20 = v48;
        v39 = v36;
        if ( v57 == v36 || (v46 = v48, v47 = v36, v40 = sub_8D97D0(v36, v57, 0, v37, v38), v39 = v47, v20 = v46, v40) )
        {
          while ( *(_BYTE *)(v39 + 140) == 12 )
            v39 = *(_QWORD *)(v39 + 160);
          v45 = *(_QWORD *)(v32 + 192);
          if ( v45 < 0 || v50 <= v45 || *(_QWORD *)(v39 + 128) > v50 - v45 )
            return 1;
          goto LABEL_22;
        }
        v19 = v47;
      }
    }
    return 1;
  }
  v19 = *(_QWORD *)(v60[0] + 128LL);
LABEL_22:
  v55 = v20;
  if ( !v19 )
    return 1;
  if ( !(unsigned int)sub_8D3410(v19) )
    return 1;
  v3 = sub_8D23B0(v19);
  if ( v3 )
    return 1;
  v21 = v55;
  for ( j = v19; *(_BYTE *)(j + 140) == 12; j = *(_QWORD *)(j + 160) )
    ;
  v49 = v21;
  for ( k = sub_8D46C0(v51); *(_BYTE *)(k + 140) == 12; k = *(_QWORD *)(k + 160) )
    ;
  v23 = j;
  v52 = k;
  v24 = sub_8D4050(j);
  v27 = v52;
  v28 = v49;
  for ( m = v24; *(_BYTE *)(m + 140) == 12; m = *(_QWORD *)(m + 160) )
    ;
  if ( m != v52 )
  {
    v23 = v52;
    v30 = sub_8D97D0(v52, m, 0, v26, v52);
    v28 = v49;
    if ( !v30 )
      return 1;
  }
  v53 = v28;
  v60[0] = sub_724DC0(v23, m, v25, v26, v27, v28);
  *v53 = 5 * (*(_BYTE *)(v6 + 56) == 92) + 170;
  if ( *(_BYTE *)(v6 + 56) != 51 )
    goto LABEL_34;
  if ( (unsigned int)sub_620E90(v13) )
  {
    v41 = (_DWORD *)v60[0];
    v42 = (_DWORD *)v13;
    for ( n = 52; n; --n )
      *v41++ = *v42++;
    v13 = v60[0];
    sub_621710((__int16 *)(v60[0] + 176LL), &v59);
    if ( !v59 )
    {
LABEL_34:
      if ( (int)sub_6210B0(v13, 0) >= 0 )
      {
        if ( v58 )
        {
          if ( !(unsigned int)sub_6210B0(v13, 0) )
          {
            *a2 = 1;
            v3 = 1;
          }
        }
        else if ( dword_4D047EC && (unsigned int)sub_8D4070(v19)
               || *(char *)(j + 168) < 0
               || (v31 = *(_QWORD *)(j + 176), v31 <= 1) )
        {
          v3 = 1;
        }
        else
        {
          v44 = sub_621100(v13, v31);
          v3 = v44 <= 0;
          *a2 = v44 == 0;
        }
      }
    }
  }
  sub_724E30(v60);
  return v3;
}
