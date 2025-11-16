// Function: sub_E77860
// Address: 0xe77860
//
__int64 __fastcall sub_E77860(__int64 a1, int a2, __int64 a3, unsigned __int64 a4, __int64 *a5)
{
  __int64 v5; // r10
  unsigned __int64 v7; // r15
  unsigned __int64 v9; // rax
  __int64 v10; // rsi
  __int64 v11; // r9
  __int64 v12; // rax
  unsigned __int64 v13; // rdi
  unsigned __int64 v14; // r8
  unsigned __int64 v15; // rdx
  unsigned __int64 v16; // r12
  __int64 v17; // r8
  unsigned __int64 v18; // rcx
  int v19; // eax
  __int64 v20; // rbx
  unsigned __int64 v21; // r13
  unsigned __int64 v22; // r15
  char v23; // dl
  __int64 v24; // rcx
  unsigned __int64 v25; // rax
  __int64 v26; // rdi
  _BYTE *v27; // rcx
  int v28; // esi
  __int64 v29; // r12
  __int64 v30; // r13
  __int64 v31; // r9
  _BYTE *v32; // rax
  _BYTE *v33; // rsi
  char *k; // rcx
  __int64 v35; // rdi
  unsigned __int64 v36; // rax
  unsigned __int64 v37; // rdx
  __int64 result; // rax
  char v39; // cl
  _BYTE *v40; // rdx
  __int64 v41; // rsi
  char v42; // di
  __int64 v43; // rbx
  __int64 v44; // rbx
  _BYTE *v45; // rcx
  char *v46; // rdx
  char v47; // si
  __int64 v48; // rdx
  __int64 v49; // rax
  unsigned __int64 v50; // rdx
  __int64 v51; // rdx
  unsigned __int64 v52; // rdx
  __int64 v53; // rax
  __int64 v54; // rcx
  char v55; // dl
  __int64 v56; // rcx
  unsigned __int64 v57; // rax
  __int64 v58; // rdi
  _BYTE *i; // rcx
  __int64 v60; // r12
  __int64 v61; // rbx
  _BYTE *v62; // rax
  __int64 v63; // rax
  _BYTE *v64; // rcx
  char *j; // rsi
  __int64 v66; // [rsp+0h] [rbp-60h]
  __int64 v67; // [rsp+0h] [rbp-60h]
  __int64 v68; // [rsp+8h] [rbp-58h]
  __int64 v69; // [rsp+8h] [rbp-58h]
  unsigned __int8 v70; // [rsp+17h] [rbp-49h]
  char v71; // [rsp+17h] [rbp-49h]
  unsigned __int8 v72; // [rsp+17h] [rbp-49h]
  char v73; // [rsp+19h] [rbp-47h]
  _BYTE v74[64]; // [rsp+20h] [rbp-40h] BYREF

  v5 = (unsigned __int8)a2;
  v7 = BYTE2(a2);
  v9 = (255 - (unsigned __int64)(unsigned __int8)a2) / BYTE2(a2);
  v73 = BYTE1(a2);
  v10 = a5[1];
  v11 = v9;
  v12 = *(_QWORD *)(a1 + 152);
  v13 = a5[2];
  v14 = *(unsigned int *)(v12 + 28);
  v15 = v10 + 1;
  v16 = a4 / v14;
  if ( a3 == 0x7FFFFFFFFFFFFFFFLL )
  {
    if ( v16 == v11 )
    {
      if ( v15 > v13 )
      {
        sub_C8D290((__int64)a5, a5 + 3, v15, 1u, v14, v11);
        v10 = a5[1];
      }
      *(_BYTE *)(*a5 + v10) = 8;
      v63 = a5[1];
      v10 = v63 + 1;
      v15 = v63 + 2;
      a5[1] = v63 + 1;
      goto LABEL_48;
    }
    if ( a4 < v14 )
    {
LABEL_48:
      if ( v15 > a5[2] )
      {
        sub_C8D290((__int64)a5, a5 + 3, v15, 1u, v14, v11);
        v10 = a5[1];
      }
      *(_BYTE *)(*a5 + v10) = 0;
      v48 = a5[1];
      v49 = v48 + 1;
      v50 = v48 + 2;
      a5[1] = v49;
      if ( v50 > a5[2] )
      {
        sub_C8D290((__int64)a5, a5 + 3, v50, 1u, v14, v11);
        v49 = a5[1];
      }
      *(_BYTE *)(*a5 + v49) = 1;
      v51 = a5[1];
      result = v51 + 1;
      v52 = v51 + 2;
      a5[1] = result;
      if ( v52 > a5[2] )
      {
        sub_C8D290((__int64)a5, a5 + 3, v52, 1u, v14, v11);
        result = a5[1];
      }
      *(_BYTE *)(*a5 + result) = 1;
      ++a5[1];
      return result;
    }
    if ( v15 > v13 )
    {
      sub_C8D290((__int64)a5, a5 + 3, v15, 1u, v14, v11);
      v10 = a5[1];
    }
    v55 = v16 & 0x7F;
    *(_BYTE *)(*a5 + v10) = 2;
    v56 = a5[1];
    v57 = v16 >> 7;
    v58 = v56 + 1;
    a5[1] = v56 + 1;
    if ( v16 >> 7 )
    {
      for ( i = v74; ; ++i )
      {
        *i = v55 | 0x80;
        v55 = v57 & 0x7F;
        v57 >>= 7;
        if ( !v57 )
          break;
      }
      i[1] = v55;
      v60 = (_DWORD)i + 2 - (unsigned int)v74;
      v61 = v60;
      v14 = v60 + v58;
      if ( v60 + v58 <= (unsigned __int64)a5[2] )
        goto LABEL_71;
    }
    else
    {
      v74[0] = v16 & 0x7F;
      v14 = v56 + 2;
      if ( a5[2] >= (unsigned __int64)(v56 + 2) )
      {
        v61 = 1;
        v60 = 1;
        v62 = (_BYTE *)(v58 + *a5);
        goto LABEL_78;
      }
      v60 = 1;
      v61 = 1;
    }
    sub_C8D290((__int64)a5, a5 + 3, v14, 1u, v14, v11);
    v58 = a5[1];
LABEL_71:
    v62 = (_BYTE *)(v58 + *a5);
    if ( !v60 )
    {
LABEL_82:
      v10 = v58 + v61;
      a5[1] = v58 + v61;
      v15 = v58 + v61 + 1;
      goto LABEL_48;
    }
    v55 = v74[0];
LABEL_78:
    v64 = &v62[v60];
    for ( j = v74; ; v55 = *j )
    {
      *v62++ = v55;
      ++j;
      if ( v62 == v64 )
        break;
    }
    v58 = a5[1];
    goto LABEL_82;
  }
  v17 = v73;
  v18 = a3 - v73;
  v19 = v73;
  if ( v18 < v7 )
  {
    v17 = 0;
    if ( v18 + v5 <= 0xFF )
    {
      if ( v16 | a3 )
        goto LABEL_5;
LABEL_38:
      if ( v15 > v13 )
      {
        sub_C8D290((__int64)a5, a5 + 3, v15, 1u, v17, v11);
        v10 = a5[1];
      }
      result = *a5;
      *(_BYTE *)(*a5 + v10) = 1;
      ++a5[1];
      return result;
    }
  }
  if ( v15 > v13 )
  {
    v66 = v11;
    v68 = v5;
    sub_C8D290((__int64)a5, a5 + 3, v15, 1u, v17, v11);
    v10 = a5[1];
    v11 = v66;
    v5 = v68;
    v19 = v73;
  }
  v39 = a3 & 0x7F;
  *(_BYTE *)(*a5 + v10) = 3;
  v40 = v74;
  v41 = a5[1] + 1;
  v42 = a3;
  v43 = a3 >> 7;
  a5[1] = v41;
  if ( v43 )
    goto LABEL_29;
  while ( (v42 & 0x40) != 0 )
  {
    while ( 1 )
    {
      ++v40;
      v42 = v43;
      *(v40 - 1) = v39 | 0x80;
      v39 = v43 & 0x7F;
      v43 >>= 7;
      if ( !v43 )
        break;
LABEL_29:
      if ( v43 == -1 && (v42 & 0x40) != 0 )
        goto LABEL_31;
    }
  }
LABEL_31:
  *v40 = v39;
  v44 = (_DWORD)v40 + 1 - (unsigned int)v74;
  if ( v44 + v41 > (unsigned __int64)a5[2] )
  {
    v67 = v11;
    v69 = v5;
    v71 = v19;
    sub_C8D290((__int64)a5, a5 + 3, v44 + v41, 1u, v17, v11);
    v41 = a5[1];
    v11 = v67;
    v5 = v69;
    v19 = v71;
  }
  v45 = (_BYTE *)(v41 + *a5);
  if ( v44 )
  {
    v46 = v74;
    do
    {
      v47 = *v46++;
      *v45++ = v47;
    }
    while ( v46 != &v74[v44] );
    v41 = a5[1];
  }
  v10 = v44 + v41;
  v13 = a5[2];
  a5[1] = v10;
  v18 = -v19;
  v17 = 1;
  v15 = v10 + 1;
  if ( !v16 )
    goto LABEL_38;
LABEL_5:
  v20 = v18 + v5;
  if ( v11 + 256 <= v16 )
    goto LABEL_8;
  v21 = v20 + v7 * v16;
  if ( v21 <= 0xFF )
  {
    if ( v15 > v13 )
    {
      sub_C8D290((__int64)a5, a5 + 3, v15, 1u, v17, v11);
      v10 = a5[1];
    }
    result = *a5;
    *(_BYTE *)(*a5 + v10) = v21;
    ++a5[1];
  }
  else
  {
    v22 = v20 + (v16 - v11) * v7;
    if ( v22 > 0xFF )
    {
LABEL_8:
      if ( v15 > v13 )
      {
        v72 = v17;
        sub_C8D290((__int64)a5, a5 + 3, v15, 1u, v17, v11);
        v10 = a5[1];
        v17 = v72;
      }
      v23 = v16 & 0x7F;
      *(_BYTE *)(*a5 + v10) = 2;
      v24 = a5[1];
      v25 = v16 >> 7;
      v26 = v24 + 1;
      a5[1] = v24 + 1;
      if ( v16 >> 7 )
      {
        v27 = v74;
        do
        {
          v28 = (int)v27++;
          *(v27 - 1) = v23 | 0x80;
          v23 = v25 & 0x7F;
          v25 >>= 7;
        }
        while ( v25 );
        *v27 = v23;
        v29 = v28 + 2 - (unsigned int)v74;
        v30 = v29;
        v31 = v29 + v26;
        if ( v29 + v26 <= (unsigned __int64)a5[2] )
          goto LABEL_14;
      }
      else
      {
        v74[0] = v16 & 0x7F;
        v31 = v24 + 2;
        if ( a5[2] >= (unsigned __int64)(v24 + 2) )
        {
          v30 = 1;
          v29 = 1;
          v32 = (_BYTE *)(v26 + *a5);
          goto LABEL_16;
        }
        v29 = 1;
        v30 = 1;
      }
      v70 = v17;
      sub_C8D290((__int64)a5, a5 + 3, v31, 1u, v17, v31);
      v26 = a5[1];
      v17 = v70;
LABEL_14:
      v32 = (_BYTE *)(v26 + *a5);
      if ( !v29 )
        goto LABEL_20;
      v23 = v74[0];
LABEL_16:
      v33 = &v32[v29];
      for ( k = v74; ; v23 = *k )
      {
        *v32++ = v23;
        ++k;
        if ( v32 == v33 )
          break;
      }
      v26 = a5[1];
LABEL_20:
      v35 = v30 + v26;
      v36 = a5[2];
      a5[1] = v35;
      v37 = v35 + 1;
      if ( (_BYTE)v17 )
      {
        if ( v36 < v37 )
        {
          sub_C8D290((__int64)a5, a5 + 3, v37, 1u, v17, v31);
          v35 = a5[1];
        }
        result = *a5;
        *(_BYTE *)(*a5 + v35) = 1;
        ++a5[1];
      }
      else
      {
        if ( v36 < v37 )
        {
          sub_C8D290((__int64)a5, a5 + 3, v37, 1u, v17, v31);
          v35 = a5[1];
        }
        result = *a5;
        *(_BYTE *)(*a5 + v35) = v20;
        ++a5[1];
      }
      return result;
    }
    if ( v15 > v13 )
    {
      sub_C8D290((__int64)a5, a5 + 3, v15, 1u, v17, v11);
      v10 = a5[1];
    }
    *(_BYTE *)(*a5 + v10) = 8;
    v53 = a5[1];
    v54 = v53 + 1;
    a5[1] = v53 + 1;
    if ( v53 + 2 > (unsigned __int64)a5[2] )
    {
      sub_C8D290((__int64)a5, a5 + 3, v53 + 2, 1u, v17, v11);
      v54 = a5[1];
    }
    result = *a5;
    *(_BYTE *)(*a5 + v54) = v22;
    ++a5[1];
  }
  return result;
}
