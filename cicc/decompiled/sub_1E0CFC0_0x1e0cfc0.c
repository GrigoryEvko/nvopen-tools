// Function: sub_1E0CFC0
// Address: 0x1e0cfc0
//
__int64 __fastcall sub_1E0CFC0(_QWORD *a1, __int64 a2)
{
  _BYTE *v3; // r8
  _DWORD *v4; // r13
  _DWORD *v5; // r9
  _DWORD *v6; // r10
  __int64 v7; // rsi
  __int64 v8; // r12
  __int64 v9; // rax
  char *v10; // r14
  char *v11; // r10
  signed __int64 v12; // r9
  unsigned __int64 v13; // r11
  unsigned __int64 v14; // rdx
  unsigned int v15; // r12d
  __int64 v16; // r13
  __int64 v17; // rax
  const void *v18; // r14
  signed __int64 v19; // rdx
  size_t v20; // rcx
  signed __int64 v21; // r8
  char *v22; // r8
  __int64 v23; // r9
  _BYTE *v24; // rsi
  char *v26; // rax
  __int64 v27; // rsi
  unsigned __int64 v28; // rax
  bool v29; // cf
  unsigned __int64 v30; // r11
  _BYTE *v31; // r15
  char *v32; // r10
  size_t v33; // rcx
  size_t v34; // rax
  size_t v35; // r13
  const void *v36; // rsi
  __int64 v37; // r11
  __int64 v38; // rax
  char *dest; // [rsp+0h] [rbp-70h]
  size_t v40; // [rsp+8h] [rbp-68h]
  size_t n; // [rsp+10h] [rbp-60h]
  size_t na; // [rsp+10h] [rbp-60h]
  size_t nb; // [rsp+10h] [rbp-60h]
  unsigned __int64 v44; // [rsp+18h] [rbp-58h]
  char *v45; // [rsp+18h] [rbp-58h]
  char *v46; // [rsp+18h] [rbp-58h]
  char *v47; // [rsp+18h] [rbp-58h]
  signed __int64 v48; // [rsp+18h] [rbp-58h]
  __int64 v49; // [rsp+20h] [rbp-50h]
  unsigned __int64 v50; // [rsp+20h] [rbp-50h]
  signed __int64 v51; // [rsp+20h] [rbp-50h]
  char *v52; // [rsp+20h] [rbp-50h]
  void *v53; // [rsp+28h] [rbp-48h]
  char *v54; // [rsp+28h] [rbp-48h]
  void *v55; // [rsp+28h] [rbp-48h]
  char *v56; // [rsp+28h] [rbp-48h]
  char *v57; // [rsp+28h] [rbp-48h]
  char *v58; // [rsp+28h] [rbp-48h]
  _DWORD v59[13]; // [rsp+3Ch] [rbp-34h] BYREF

  v3 = *(_BYTE **)(a2 + 8);
  v4 = *(_DWORD **)a2;
  v5 = (_DWORD *)a1[72];
  v6 = (_DWORD *)a1[73];
  v7 = (__int64)&v3[-*(_QWORD *)a2] >> 2;
  if ( v5 != v6 )
  {
    while ( 2 )
    {
      LODWORD(v8) = *v5;
      LODWORD(v9) = v7;
      while ( (_DWORD)v8 )
      {
        if ( !(_DWORD)v9 )
          return (unsigned int)~(_DWORD)v8;
        v9 = (unsigned int)(v9 - 1);
        v8 = (unsigned int)(v8 - 1);
        if ( *(_DWORD *)(a1[69] + 4 * v8) != v4[v9] )
          goto LABEL_6;
      }
      if ( !(_DWORD)v9 )
        return (unsigned int)~(_DWORD)v8;
LABEL_6:
      if ( v6 != ++v5 )
        continue;
      break;
    }
  }
  v10 = (char *)a1[70];
  v11 = (char *)a1[69];
  v12 = v10 - v11;
  v13 = (v10 - v11) >> 2;
  v14 = v7 + v13 + 1;
  v15 = ~(_DWORD)v13;
  if ( v14 > 0x1FFFFFFFFFFFFFFFLL )
    sub_4262D8((__int64)"vector::reserve");
  if ( v14 > (__int64)(a1[71] - (_QWORD)v11) >> 2 )
  {
    v16 = 4 * v14;
    if ( v7 + v13 == -1 )
    {
      v19 = a1[70] - (_QWORD)v11;
      v18 = (const void *)a1[69];
      v11 = 0;
      if ( v12 <= 0 )
        goto LABEL_11;
    }
    else
    {
      v49 = (__int64)(a1[70] - (_QWORD)v11) >> 2;
      v53 = (void *)(a1[70] - (_QWORD)v11);
      v17 = sub_22077B0(4 * v14);
      v18 = (const void *)a1[69];
      v12 = (signed __int64)v53;
      v13 = v49;
      v11 = (char *)v17;
      v19 = a1[70] - (_QWORD)v18;
      if ( v19 <= 0 )
      {
LABEL_11:
        if ( !v18 )
        {
LABEL_12:
          v10 = &v11[v12];
          a1[69] = v11;
          a1[70] = &v11[v12];
          a1[71] = &v11[v16];
          v4 = *(_DWORD **)a2;
          v3 = *(_BYTE **)(a2 + 8);
          goto LABEL_13;
        }
        v27 = a1[71] - (_QWORD)v18;
LABEL_29:
        v44 = v13;
        v51 = v12;
        v56 = v11;
        j_j___libc_free_0(v18, v27);
        v13 = v44;
        v12 = v51;
        v11 = v56;
        goto LABEL_12;
      }
    }
    v50 = v13;
    v55 = (void *)v12;
    v26 = (char *)memmove(v11, v18, v19);
    v12 = (signed __int64)v55;
    v13 = v50;
    v11 = v26;
    v27 = a1[71] - (_QWORD)v18;
    goto LABEL_29;
  }
LABEL_13:
  if ( v4 != (_DWORD *)v3 )
  {
    v20 = a1[71];
    v21 = v3 - (_BYTE *)v4;
    if ( v20 - (unsigned __int64)v10 >= v21 )
    {
      v54 = (char *)v21;
      memmove(v10, v4, v21);
      v22 = &v54[a1[70]];
      a1[70] = v22;
      v10 = v22;
      v12 = (signed __int64)&v22[-a1[69]];
      goto LABEL_16;
    }
    v28 = v21 >> 2;
    if ( v21 >> 2 > 0x1FFFFFFFFFFFFFFFLL - v13 )
      sub_4262D8((__int64)"vector::_M_range_insert");
    if ( v13 >= v28 )
      v28 = v13;
    v29 = __CFADD__(v28, v13);
    v30 = v28 + v13;
    if ( v29 )
    {
      v37 = 0x7FFFFFFFFFFFFFFCLL;
    }
    else
    {
      if ( !v30 )
      {
        v52 = 0;
        v31 = 0;
        goto LABEL_36;
      }
      if ( v30 > 0x1FFFFFFFFFFFFFFFLL )
        v30 = 0x1FFFFFFFFFFFFFFFLL;
      v37 = 4 * v30;
    }
    v48 = v21;
    v58 = (char *)v37;
    v38 = sub_22077B0(v37);
    v21 = v48;
    v11 = (char *)a1[69];
    v31 = (_BYTE *)v38;
    v20 = a1[71];
    v52 = &v58[v38];
    v12 = v10 - v11;
LABEL_36:
    v57 = &v31[v12 + v21];
    if ( v11 == v10 )
    {
      v36 = v4;
      nb = v20;
      v47 = v11;
      v35 = 0;
      memcpy(&v31[v12], v36, v21);
      v32 = v47;
      v33 = nb;
      v34 = a1[70] - (_QWORD)v47;
      if ( (char *)a1[70] == v47 )
      {
LABEL_39:
        v10 = &v57[v35];
        if ( !v32 )
        {
LABEL_40:
          a1[69] = v31;
          a1[70] = v10;
          v12 = v10 - v31;
          a1[71] = v52;
          goto LABEL_16;
        }
LABEL_44:
        j_j___libc_free_0(v32, v33 - (_QWORD)v32);
        goto LABEL_40;
      }
    }
    else
    {
      v40 = v20;
      v45 = v11;
      dest = &v31[v12];
      n = v21;
      memmove(v31, v11, v12);
      memcpy(dest, v4, n);
      v32 = v45;
      v33 = v40;
      v34 = a1[70] - (_QWORD)v10;
      if ( (char *)a1[70] == v10 )
      {
        v10 = v57;
        goto LABEL_44;
      }
    }
    na = v33;
    v46 = v32;
    v35 = v34;
    memcpy(v57, v10, v34);
    v33 = na;
    v32 = v46;
    goto LABEL_39;
  }
LABEL_16:
  v23 = v12 >> 2;
  v24 = (_BYTE *)a1[73];
  v59[0] = v23;
  if ( v24 == (_BYTE *)a1[74] )
  {
    sub_C88AB0((__int64)(a1 + 72), v24, v59);
    v10 = (char *)a1[70];
  }
  else
  {
    if ( v24 )
    {
      *(_DWORD *)v24 = v23;
      v24 = (_BYTE *)a1[73];
      v10 = (char *)a1[70];
    }
    a1[73] = v24 + 4;
  }
  v59[0] = 0;
  if ( (char *)a1[71] == v10 )
  {
    sub_C88AB0((__int64)(a1 + 69), v10, v59);
  }
  else
  {
    if ( v10 )
    {
      *(_DWORD *)v10 = 0;
      v10 = (char *)a1[70];
    }
    a1[70] = v10 + 4;
  }
  return v15;
}
