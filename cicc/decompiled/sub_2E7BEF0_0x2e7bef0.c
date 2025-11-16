// Function: sub_2E7BEF0
// Address: 0x2e7bef0
//
__int64 __fastcall sub_2E7BEF0(_QWORD *a1, _DWORD *a2, __int64 a3)
{
  _DWORD *v5; // r8
  _DWORD *v6; // rdi
  __int64 v7; // r12
  __int64 v8; // rax
  char *v9; // r14
  char *v10; // r10
  signed __int64 v11; // r9
  unsigned __int64 v12; // r11
  unsigned __int64 v13; // rdx
  unsigned int v14; // r12d
  __int64 v15; // rcx
  __int64 v16; // rax
  const void *v17; // r14
  signed __int64 v18; // rdx
  signed __int64 v19; // r15
  char *v20; // r15
  __int64 v21; // r9
  _BYTE *v22; // rsi
  char *v24; // rax
  unsigned __int64 v25; // rax
  bool v26; // cf
  unsigned __int64 v27; // r11
  unsigned __int64 v28; // r10
  size_t v29; // rax
  size_t v30; // r13
  unsigned __int64 v31; // r11
  __int64 v32; // rax
  char *v33; // [rsp+8h] [rbp-68h]
  unsigned __int64 v34; // [rsp+10h] [rbp-60h]
  char *v35; // [rsp+10h] [rbp-60h]
  unsigned __int64 v36; // [rsp+10h] [rbp-60h]
  char *v37; // [rsp+10h] [rbp-60h]
  __int64 v38; // [rsp+18h] [rbp-58h]
  unsigned __int64 v39; // [rsp+18h] [rbp-58h]
  __int64 v40; // [rsp+18h] [rbp-58h]
  char *v41; // [rsp+18h] [rbp-58h]
  void *desta; // [rsp+20h] [rbp-50h]
  void *destb; // [rsp+20h] [rbp-50h]
  void *destc; // [rsp+20h] [rbp-50h]
  _BYTE *dest; // [rsp+20h] [rbp-50h]
  void *v46; // [rsp+28h] [rbp-48h]
  void *v47; // [rsp+28h] [rbp-48h]
  char *v48; // [rsp+28h] [rbp-48h]
  char *v49; // [rsp+28h] [rbp-48h]
  char *v50; // [rsp+28h] [rbp-48h]
  _DWORD v51[13]; // [rsp+3Ch] [rbp-34h] BYREF

  v5 = (_DWORD *)a1[81];
  v6 = (_DWORD *)a1[80];
  if ( v6 != v5 )
  {
    while ( 2 )
    {
      LODWORD(v7) = *v6;
      LODWORD(v8) = a3;
      while ( (_DWORD)v7 )
      {
        if ( !(_DWORD)v8 )
          return (unsigned int)~(_DWORD)v7;
        v8 = (unsigned int)(v8 - 1);
        v7 = (unsigned int)(v7 - 1);
        if ( *(_DWORD *)(a1[77] + 4 * v7) != a2[v8] )
          goto LABEL_6;
      }
      if ( !(_DWORD)v8 )
        return (unsigned int)~(_DWORD)v7;
LABEL_6:
      if ( v5 != ++v6 )
        continue;
      break;
    }
  }
  v9 = (char *)a1[78];
  v10 = (char *)a1[77];
  v11 = v9 - v10;
  v12 = (v9 - v10) >> 2;
  v13 = a3 + v12 + 1;
  v14 = ~(_DWORD)v12;
  if ( v13 > 0x1FFFFFFFFFFFFFFFLL )
    sub_4262D8((__int64)"vector::reserve");
  if ( v13 > (__int64)(a1[79] - (_QWORD)v10) >> 2 )
  {
    v15 = 4 * v13;
    if ( a3 + v12 == -1 )
    {
      v18 = a1[78] - (_QWORD)v10;
      v17 = (const void *)a1[77];
      v10 = 0;
      if ( v11 <= 0 )
      {
LABEL_11:
        if ( !v17 )
        {
LABEL_12:
          v9 = &v10[v11];
          a1[77] = v10;
          a1[78] = &v10[v11];
          a1[79] = &v10[v15];
          goto LABEL_13;
        }
LABEL_29:
        v34 = v12;
        v40 = v15;
        destc = (void *)v11;
        v48 = v10;
        j_j___libc_free_0((unsigned __int64)v17);
        v12 = v34;
        v15 = v40;
        v11 = (signed __int64)destc;
        v10 = v48;
        goto LABEL_12;
      }
    }
    else
    {
      v38 = (__int64)(a1[78] - (_QWORD)v10) >> 2;
      desta = (void *)(a1[78] - (_QWORD)v10);
      v46 = (void *)(4 * v13);
      v16 = sub_22077B0(4 * v13);
      v17 = (const void *)a1[77];
      v15 = (__int64)v46;
      v11 = (signed __int64)desta;
      v10 = (char *)v16;
      v18 = a1[78] - (_QWORD)v17;
      v12 = v38;
      if ( v18 <= 0 )
        goto LABEL_11;
    }
    v39 = v12;
    destb = (void *)v15;
    v47 = (void *)v11;
    v24 = (char *)memmove(v10, v17, v18);
    v11 = (signed __int64)v47;
    v15 = (__int64)destb;
    v12 = v39;
    v10 = v24;
    goto LABEL_29;
  }
LABEL_13:
  v19 = 4 * a3;
  if ( v19 )
  {
    if ( (unsigned __int64)v19 <= a1[79] - (_QWORD)v9 )
    {
      memmove(v9, a2, v19);
      v20 = (char *)(a1[78] + v19);
      a1[78] = v20;
      v9 = v20;
      v11 = (signed __int64)&v20[-a1[77]];
      goto LABEL_16;
    }
    v25 = v19 >> 2;
    if ( v19 >> 2 > 0x1FFFFFFFFFFFFFFFLL - v12 )
      sub_4262D8((__int64)"vector::_M_range_insert");
    if ( v12 >= v25 )
      v25 = v12;
    v26 = __CFADD__(v25, v12);
    v27 = v25 + v12;
    if ( v26 )
    {
      v31 = 0x7FFFFFFFFFFFFFFCLL;
    }
    else
    {
      if ( !v27 )
      {
        v41 = 0;
        dest = 0;
        goto LABEL_36;
      }
      if ( v27 > 0x1FFFFFFFFFFFFFFFLL )
        v27 = 0x1FFFFFFFFFFFFFFFLL;
      v31 = 4 * v27;
    }
    v50 = (char *)v31;
    v32 = sub_22077B0(v31);
    v10 = (char *)a1[77];
    dest = (_BYTE *)v32;
    v11 = v9 - v10;
    v41 = &v50[v32];
LABEL_36:
    v49 = &dest[v11 + v19];
    if ( v9 == v10 )
    {
      v37 = v10;
      v30 = 0;
      memcpy(&dest[v11], a2, v19);
      v28 = (unsigned __int64)v37;
      v29 = a1[78] - (_QWORD)v9;
      if ( v9 == (char *)a1[78] )
      {
LABEL_39:
        v9 = &v49[v30];
        if ( !v28 )
        {
LABEL_40:
          a1[78] = v9;
          a1[77] = dest;
          v11 = v9 - dest;
          a1[79] = v41;
          goto LABEL_16;
        }
LABEL_44:
        j_j___libc_free_0(v28);
        goto LABEL_40;
      }
    }
    else
    {
      v35 = v10;
      v33 = &dest[v11];
      memmove(dest, v10, v11);
      memcpy(v33, a2, v19);
      v28 = (unsigned __int64)v35;
      v29 = a1[78] - (_QWORD)v9;
      if ( (char *)a1[78] == v9 )
      {
        v9 = v49;
        goto LABEL_44;
      }
    }
    v36 = v28;
    v30 = v29;
    memcpy(v49, v9, v29);
    v28 = v36;
    goto LABEL_39;
  }
LABEL_16:
  v21 = v11 >> 2;
  v22 = (_BYTE *)a1[81];
  v51[0] = v21;
  if ( v22 == (_BYTE *)a1[82] )
  {
    sub_C88AB0((__int64)(a1 + 80), v22, v51);
    v9 = (char *)a1[78];
  }
  else
  {
    if ( v22 )
    {
      *(_DWORD *)v22 = v21;
      v22 = (_BYTE *)a1[81];
      v9 = (char *)a1[78];
    }
    a1[81] = v22 + 4;
  }
  v51[0] = 0;
  if ( (char *)a1[79] == v9 )
  {
    sub_C88AB0((__int64)(a1 + 77), v9, v51);
  }
  else
  {
    if ( v9 )
    {
      *(_DWORD *)v9 = 0;
      v9 = (char *)a1[78];
    }
    a1[78] = v9 + 4;
  }
  return v14;
}
