// Function: sub_6F35D0
// Address: 0x6f35d0
//
_DWORD *__fastcall sub_6F35D0(__int64 a1, _QWORD *a2)
{
  _BYTE *n; // rsi
  __int64 v4; // r13
  bool v5; // zf
  _DWORD *result; // rax
  __int64 v7; // rbx
  unsigned int v8; // eax
  __int64 v9; // rdx
  __int64 m; // r8
  int v11; // r9d
  __int64 v12; // rcx
  __int64 v13; // rbx
  unsigned int k; // r15d
  __int64 *v15; // rbx
  __int64 v16; // r12
  __int64 v17; // r15
  int v18; // eax
  __int64 v19; // rax
  __int64 v20; // r12
  __int64 v21; // rax
  __int64 v22; // rdi
  char i; // al
  __int64 *v24; // rax
  int v25; // eax
  __int64 j; // r13
  __int64 v27; // rax
  __int64 v28; // rdi
  __int64 ii; // rbx
  __int64 v30; // rax
  __int64 v31; // rcx
  __int64 v32; // rax
  __int64 v33; // r12
  int v34; // eax
  int v35; // esi
  unsigned int v36; // eax
  unsigned int v37; // edx
  __int64 v38; // rdi
  __int64 *v39; // r9
  unsigned int v40; // edx
  __int64 *v41; // rax
  __int64 v42; // rdx
  __int64 *v43; // rdi
  int v44; // eax
  __int64 v45; // [rsp+0h] [rbp-130h]
  int v46; // [rsp+Ch] [rbp-124h]
  __int64 v47; // [rsp+10h] [rbp-120h]
  int v48; // [rsp+10h] [rbp-120h]
  __int64 v49; // [rsp+18h] [rbp-118h]
  __int64 v50; // [rsp+18h] [rbp-118h]
  __int64 v51; // [rsp+20h] [rbp-110h]
  _BYTE v52[128]; // [rsp+30h] [rbp-100h] BYREF
  __int64 v53; // [rsp+B0h] [rbp-80h]
  __int64 v54; // [rsp+C0h] [rbp-70h]
  char v55; // [rsp+DDh] [rbp-53h]
  char v56; // [rsp+E0h] [rbp-50h]
  char v57; // [rsp+E1h] [rbp-4Fh]
  __int64 v58; // [rsp+E8h] [rbp-48h]

  n = v52;
  v4 = *(_QWORD *)(a1 + 88);
  sub_72A510(v4, v52);
  if ( v55 == 12 && v56 == 1 && (v57 & 0x10) != 0 )
  {
    v19 = sub_72E9A0(v4);
    v57 &= ~0x10u;
    v58 = v19;
  }
  sub_72A140(v52);
  v5 = *(_BYTE *)(v4 + 173) == 0;
  v54 = 0;
  if ( v5 )
    return (_DWORD *)sub_6E6260(a2);
  if ( (unsigned int)sub_8D2FB0(v53) )
  {
    v20 = sub_73A720(v52);
    if ( qword_4F04C50 )
    {
      v21 = *(_QWORD *)(qword_4F04C50 + 32LL);
      if ( v21 )
      {
        if ( (*(_BYTE *)(v21 + 198) & 0x10) != 0 && (!qword_4D03C50 || *(char *)(qword_4D03C50 + 18LL) >= 0) )
        {
          v22 = *(_QWORD *)v20;
          for ( i = *(_BYTE *)(*(_QWORD *)v20 + 140LL); i == 12; i = *(_BYTE *)(v22 + 140) )
            v22 = *(_QWORD *)(v22 + 160);
          if ( i == 6 )
          {
            for ( j = sub_8D46C0(v22); *(_BYTE *)(j + 140) == 12; j = *(_QWORD *)(j + 160) )
              ;
            if ( (unsigned int)sub_8D2FF0(j, v52) || (unsigned int)sub_8D3030(j) )
              sub_6851C0(0xDE6u, &dword_4F063F8);
          }
        }
      }
    }
    v24 = (__int64 *)sub_73DDB0(v20);
    return (_DWORD *)sub_6E7150(v24, (__int64)a2);
  }
  else if ( *(_BYTE *)(v4 + 173) == 10 && (*(_BYTE *)(a1 + 81) & 0x40) != 0 )
  {
    v7 = qword_4D03A70;
    v8 = sub_72DB90(v4);
    v11 = *(_DWORD *)(v7 + 8);
    v12 = *(_QWORD *)v7;
    v13 = v11 & v8;
    for ( k = v11 & v8; ; v13 = k )
    {
      v15 = (__int64 *)(v12 + 16 * v13);
      v16 = *v15;
      if ( v4 == *v15 )
        break;
      if ( v16 && *(_BYTE *)(v16 + 173) == *(_BYTE *)(v4 + 173) )
      {
        for ( m = *(_QWORD *)(v16 + 128); *(_BYTE *)(m + 140) == 12; m = *(_QWORD *)(m + 160) )
          ;
        for ( n = *(_BYTE **)(v4 + 128); n[140] == 12; n = (_BYTE *)*((_QWORD *)n + 20) )
          ;
        if ( (_BYTE *)m == n
          || (v46 = v11, v47 = v12, v49 = m, v25 = sub_8D97D0(m, n, 0, v12, m), m = v49, v12 = v47, v11 = v46, v25) )
        {
          n = (_BYTE *)v4;
          v48 = v11;
          v50 = v12;
          if ( (unsigned int)sub_6E0A40(v16, v4, m, v12, m) )
            break;
          v12 = v50;
          v11 = v48;
        }
      }
      if ( !*v15 )
        goto LABEL_47;
      k = v11 & (k + 1);
    }
    v17 = v15[1];
    if ( v17 )
      return (_DWORD *)sub_6F8E70(v17, &dword_4F063F8, &qword_4F063F0, a2, 0);
LABEL_47:
    v27 = sub_87F5F0(dword_4F07508, n, v9, v12, m);
    v28 = *(_QWORD *)(v4 + 128);
    for ( ii = v27; *(_BYTE *)(v28 + 140) == 12; v28 = *(_QWORD *)(v28 + 160) )
      ;
    v30 = sub_73C570(v28, 1, -1);
    v32 = sub_735FB0(v30, 0, 0, v31);
    *(_BYTE *)(v32 + 172) |= 8u;
    v17 = v32;
    LOWORD(v32) = *(_WORD *)(v32 + 176);
    *(_QWORD *)(v17 + 184) = v4;
    *(_WORD *)(v17 + 176) = v32 & 0xDF | 0x120;
    sub_658080((_BYTE *)v17, 1);
    sub_877D80(v17, ii);
    v33 = qword_4D03A70;
    v34 = sub_72DB90(v4);
    v35 = *(_DWORD *)(v33 + 8);
    v36 = v35 & v34;
    v37 = v36 + 1;
    v38 = 16LL * v36;
    v39 = (__int64 *)(*(_QWORD *)v33 + v38);
    if ( *v39 )
    {
      while ( 1 )
      {
        v40 = v35 & v37;
        v41 = (__int64 *)(*(_QWORD *)v33 + 16LL * v40);
        if ( !*v41 )
          break;
        v37 = v40 + 1;
      }
      v51 = *v41;
      v42 = *v39;
      *v41 = *v39;
      if ( v42 )
        v41[1] = v39[1];
      *v39 = v51;
      if ( v51 )
        v39[1] = v45;
      v43 = (__int64 *)(*(_QWORD *)v33 + v38);
      *v43 = v4;
      v43[1] = v17;
    }
    else
    {
      *v39 = v4;
      v39[1] = v17;
    }
    v44 = *(_DWORD *)(v33 + 12) + 1;
    *(_DWORD *)(v33 + 12) = v44;
    if ( (unsigned int)(2 * v44) > *(_DWORD *)(v33 + 8) )
      sub_6F34D0(v33);
    *(_QWORD *)(ii + 88) = v17;
    return (_DWORD *)sub_6F8E70(v17, &dword_4F063F8, &qword_4F063F0, a2, 0);
  }
  else
  {
    result = (_DWORD *)sub_6E6A50((__int64)v52, (__int64)a2);
    if ( (*(_BYTE *)(a1 + 81) & 0x40) != 0 )
    {
      result = &dword_4F077C4;
      if ( dword_4F077C4 == 2 )
      {
        result = &unk_4F07778;
        if ( unk_4F07778 > 202001 && *(_BYTE *)(v4 + 173) == 12 )
        {
          if ( (unsigned int)sub_8D3A70(*(_QWORD *)(v4 + 128))
            || (result = (_DWORD *)sub_8DD3B0(*(_QWORD *)(v4 + 128)), (_DWORD)result) )
          {
            v18 = sub_8DBE70(*(_QWORD *)(v4 + 128));
            return (_DWORD *)sub_6F7FE0(a2, v18 == 0);
          }
        }
      }
    }
  }
  return result;
}
