// Function: sub_775C20
// Address: 0x775c20
//
__int64 __fastcall sub_775C20(__int64 a1, __int64 a2, __int64 a3, __int64 *a4, __int64 a5, __int64 a6)
{
  __int64 v7; // r12
  __int64 i; // rax
  char v9; // si
  __int64 v10; // rdi
  __int64 v11; // rax
  __int64 j; // r13
  const char *v13; // r15
  unsigned int v14; // r12d
  unsigned int v16; // edi
  __int64 v17; // r8
  unsigned int v18; // r9d
  unsigned int v19; // ecx
  __m128i *v20; // r10
  const char *v21; // rsi
  int v22; // ecx
  unsigned int v23; // r14d
  unsigned int v24; // esi
  __int64 v25; // r12
  _WORD *v26; // r12
  __m128i v27; // xmm0
  __m128i *v28; // rax
  _WORD *v29; // r13
  _WORD *v30; // rbx
  unsigned int v31; // r12d
  __int64 v32; // rax
  int v33; // ecx
  __int64 v34; // rax
  unsigned __int64 v35; // r12
  char v36; // dl
  const char **v37; // rax
  __int64 v38; // r9
  unsigned int v39; // r10d
  __int64 v40; // r8
  unsigned int v41; // eax
  __int64 v42; // rdi
  unsigned __int64 *v43; // rcx
  unsigned __int64 v44; // rsi
  __int64 v45; // rax
  int v46; // esi
  unsigned __int64 v47; // r13
  __int64 v48; // rax
  char k; // cl
  unsigned int m; // edx
  __int64 v51; // rax
  __int64 v52; // rax
  int v53; // r15d
  __int64 v54; // rax
  __int64 v55; // rax
  __int64 v56; // rcx
  int v57; // edx
  __int64 v58; // rax
  unsigned int v59; // r14d
  __int64 v60; // r12
  __int64 v61; // rax
  __int64 v62; // [rsp+8h] [rbp-68h]
  __int64 v63; // [rsp+10h] [rbp-60h]
  unsigned int v64; // [rsp+18h] [rbp-58h]
  unsigned int v66; // [rsp+2Ch] [rbp-44h]
  int v67; // [rsp+2Ch] [rbp-44h]
  __int64 v68; // [rsp+30h] [rbp-40h]
  int v69; // [rsp+30h] [rbp-40h]
  int v70; // [rsp+30h] [rbp-40h]

  v7 = *(_QWORD *)(a2 + 152);
  for ( i = *a4; *(_BYTE *)(v7 + 140) == 12; v7 = *(_QWORD *)(v7 + 160) )
    ;
  v9 = *(_BYTE *)i;
  v10 = *(_QWORD *)(i + 8);
  if ( *(_BYTE *)i == 48 )
  {
    v36 = *(_BYTE *)(v10 + 8);
    if ( v36 == 1 )
    {
      *(_BYTE *)i = 2;
      v10 = *(_QWORD *)(v10 + 32);
      v9 = 2;
      *(_QWORD *)(i + 8) = v10;
    }
    else if ( v36 == 2 )
    {
      *(_BYTE *)i = 59;
      v10 = *(_QWORD *)(v10 + 32);
      v9 = 59;
      *(_QWORD *)(i + 8) = v10;
    }
    else
    {
      if ( v36 )
        sub_721090();
      *(_BYTE *)i = 6;
      v10 = *(_QWORD *)(v10 + 32);
      v9 = 6;
      *(_QWORD *)(i + 8) = v10;
    }
  }
  v11 = sub_72A270(v10, v9);
  for ( j = *(_QWORD *)(v7 + 160); *(_BYTE *)(j + 140) == 12; j = *(_QWORD *)(j + 160) )
    ;
  if ( !v11 )
  {
    v66 = 9;
    v13 = "<invalid>";
    v64 = 9;
    goto LABEL_10;
  }
  v13 = *(const char **)(v11 + 8);
  if ( v13 )
  {
    if ( (*(_BYTE *)(v11 + 89) & 0x40) != 0 )
    {
      v13 = 0;
      v64 = strlen(0);
      v66 = v64;
    }
    else
    {
      if ( (*(_BYTE *)(v11 + 89) & 8) != 0 )
        v13 = *(const char **)(v11 + 24);
      v64 = strlen(v13);
      v66 = v64;
    }
LABEL_10:
    if ( **(_QWORD **)(j + 168) )
      goto LABEL_11;
    goto LABEL_15;
  }
  v66 = 0;
  v13 = byte_3F871B3;
  v64 = 0;
  if ( **(_QWORD **)(j + 168) )
    goto LABEL_11;
LABEL_15:
  v16 = qword_4F08388;
  v17 = qword_4F08380;
  v18 = ((unsigned __int64)v13 >> 3) & qword_4F08388;
  v19 = v18;
  v20 = (__m128i *)(qword_4F08380 + 16LL * v18);
  v21 = (const char *)v20->m128i_i64[0];
  if ( v13 == (const char *)v20->m128i_i64[0] )
  {
    v37 = (const char **)(qword_4F08380 + 16LL * v18);
LABEL_38:
    v35 = (unsigned __int64)v37[1];
    if ( v35 )
      goto LABEL_39;
  }
  else
  {
    while ( v21 )
    {
      v19 = qword_4F08388 & (v19 + 1);
      v37 = (const char **)(qword_4F08380 + 16LL * v19);
      v21 = *v37;
      if ( v13 == *v37 )
        goto LABEL_38;
    }
  }
  v22 = (2 * ((_BYTE)v66 + 1) + 9) & 7;
  v23 = ((2 * (v66 + 1)) & 0x1FFFFFFF) + 17 - v22 + 16 * (v66 + 1);
  v24 = ((2 * (v66 + 1)) & 0x1FFFFFFF) + 17 - v22;
  if ( v23 > 0x400 )
  {
    v59 = v23 + 16;
    v24 = ((2 * (v66 + 1)) & 0x1FFFFFFF) + 17 - v22;
    v60 = sub_822B10(v59);
    v61 = qword_4F083B0;
    *(_DWORD *)(v60 + 8) = v59;
    *(_QWORD *)v60 = v61;
    v16 = qword_4F08388;
    v17 = qword_4F08380;
    v18 = qword_4F08388 & ((unsigned __int64)v13 >> 3);
    *(_DWORD *)(v60 + 12) = dword_4F083B8;
    qword_4F083B0 = v60;
    v25 = v60 + 16;
    v20 = (__m128i *)(v17 + 16LL * v18);
  }
  else
  {
    v25 = qword_4F083A0;
    if ( ((2 * ((_BYTE)v66 + 1) + 17 - (_BYTE)v22 + 16 * ((_BYTE)v66 + 1)) & 7) != 0 )
      v23 = v23 + 8 - (v23 & 7);
    if ( 0x10000 - ((int)qword_4F083A0 - (int)qword_4F083A8) < v23 )
    {
      sub_772E70(&qword_4F083A0);
      v16 = qword_4F08388;
      v17 = qword_4F08380;
      v25 = qword_4F083A0;
      v18 = qword_4F08388 & ((unsigned __int64)v13 >> 3);
      v20 = (__m128i *)(qword_4F08380 + 16LL * v18);
    }
    qword_4F083A0 = v25 + v23;
  }
  v26 = (_WORD *)(v24 + v25);
  if ( v20->m128i_i64[0] )
  {
    v27 = _mm_loadu_si128(v20);
    v20->m128i_i64[0] = (__int64)v13;
    v20->m128i_i64[1] = (__int64)v26;
    do
    {
      v18 = v16 & (v18 + 1);
      v28 = (__m128i *)(v17 + 16LL * v18);
    }
    while ( v28->m128i_i64[0] );
    *v28 = v27;
  }
  else
  {
    v20->m128i_i64[0] = (__int64)v13;
    v20->m128i_i64[1] = (__int64)v26;
  }
  ++HIDWORD(qword_4F08388);
  if ( 2 * HIDWORD(qword_4F08388) > v16 )
    sub_7704A0((__int64)&qword_4F08380);
  v68 = j;
  v29 = v26;
  v63 = a6;
  v30 = v26;
  v31 = 0;
  do
  {
    v32 = v31++;
    sub_620D80(v29, v13[v32]);
    v33 = (int)v29;
    v29 += 8;
    v34 = -(((unsigned int)(v33 - (_DWORD)v30) >> 3) + 10);
    *((_BYTE *)v30 + v34) |= 1 << ((v33 - (_BYTE)v30) & 7);
  }
  while ( v31 <= v66 );
  v35 = (unsigned __int64)v30;
  j = v68;
  a6 = v63;
  *(_QWORD *)(v35 - 8) = sub_72BA30(byte_4F068B0[0]);
LABEL_39:
  v38 = qword_4F08070;
  v39 = *(_DWORD *)(a1 + 8);
  v40 = *(_QWORD *)a1;
  v41 = v39 & (v35 >> 3);
  v42 = v41;
  v43 = (unsigned __int64 *)(*(_QWORD *)a1 + 16LL * v41);
  v44 = *v43;
  if ( *v43 )
  {
    while ( v35 != v44 )
    {
      v41 = v39 & (v41 + 1);
      v42 = v41;
      v43 = (unsigned __int64 *)(v40 + 16LL * v41);
      v44 = *v43;
      if ( !*v43 )
        goto LABEL_60;
    }
    *(_QWORD *)(v40 + 16 * v42 + 8) = qword_4F08070;
  }
  else
  {
LABEL_60:
    *v43 = v35;
    v43[1] = v38;
    v70 = *(_DWORD *)(a1 + 12);
    *(_DWORD *)(a1 + 12) = v70 + 1;
    if ( v39 < 2 * (v70 + 1) )
      sub_7704A0(a1);
  }
  v45 = sub_76FF70(*(_QWORD *)(j + 160));
  if ( v45 )
  {
    v69 = 0;
    v46 = 0;
    v62 = j;
    v47 = v45;
    v67 = (v66 + 1) << 8;
    while ( 2 )
    {
      v48 = *(_QWORD *)(v47 + 120);
      for ( k = *(_BYTE *)(v48 + 140); k == 12; k = *(_BYTE *)(v48 + 140) )
        v48 = *(_QWORD *)(v48 + 160);
      for ( m = qword_4F08388 & (v47 >> 3); ; m = qword_4F08388 & (m + 1) )
      {
        v51 = qword_4F08380 + 16LL * m;
        if ( *(_QWORD *)v51 == v47 )
          break;
        if ( !*(_QWORD *)v51 )
        {
          v52 = 0;
          if ( k == 6 )
            goto LABEL_63;
          goto LABEL_53;
        }
      }
      v52 = *(unsigned int *)(v51 + 8);
      if ( k == 6 )
      {
LABEL_63:
        if ( v69 )
          goto LABEL_11;
        v56 = a5 + v52;
        *(_OWORD *)(v56 + 8) = 0;
        *(_QWORD *)v56 = v35;
        *(_BYTE *)(v56 + 8) = 72;
        *(_QWORD *)(v56 + 24) = v35;
        *(_DWORD *)(v56 + 8) = v67 | *(unsigned __int8 *)(v56 + 8);
        v69 = 1;
        v57 = 1 << ((a5 + v52 - a6) & 7);
        v58 = -(((unsigned int)(a5 + v52 - a6) >> 3) + 10);
        *(_BYTE *)(a6 + v58) |= v57;
        goto LABEL_56;
      }
LABEL_53:
      if ( v46 || k != 2 )
        goto LABEL_11;
      v53 = a5 + v52;
      sub_620D80((_WORD *)(a5 + (unsigned int)v52), v64);
      v46 = 1;
      v54 = -(((unsigned int)(v53 - a6) >> 3) + 10);
      *(_BYTE *)(a6 + v54) |= 1 << ((v53 - a6) & 7);
LABEL_56:
      v47 = sub_76FF70(*(_QWORD *)(v47 + 112));
      if ( v47 )
        continue;
      break;
    }
    if ( !(v46 ^ 1 | v69 ^ 1) )
    {
      v14 = 1;
      v55 = -(((unsigned int)(a5 - a6) >> 3) + 10);
      *(_BYTE *)(a6 + v55) |= 1 << ((a5 - a6) & 7);
      *(_QWORD *)(a6 - 8) = v62;
      return v14;
    }
  }
LABEL_11:
  v14 = 0;
  if ( (*(_BYTE *)(a1 + 132) & 0x20) == 0 )
  {
    sub_6855B0(0xD23u, (FILE *)(a1 + 112), (_QWORD *)(a1 + 96));
    sub_770D30(a1);
  }
  return v14;
}
