// Function: sub_142F320
// Address: 0x142f320
//
__int64 __fastcall sub_142F320(_DWORD *a1, const void **a2, size_t **a3)
{
  int v4; // r12d
  unsigned int v5; // r13d
  _BYTE *v7; // rbx
  _BYTE *v9; // rax
  _BYTE *v10; // rsi
  size_t v11; // rdx
  signed __int64 v12; // r13
  _DWORD *v13; // rax
  _BYTE *v14; // rsi
  size_t *v15; // rbx
  size_t v16; // r13
  _BYTE *v17; // rax
  size_t v18; // r15
  __int64 v19; // rax
  _DWORD *v20; // r12
  _BYTE *v21; // rax
  _BYTE *v22; // r11
  unsigned __int64 v23; // r8
  __int64 v24; // rax
  size_t v25; // r10
  int v26; // r13d
  _DWORD *v27; // rax
  _BYTE *v28; // rax
  size_t v29; // r15
  signed __int64 v30; // r12
  __int64 v31; // rax
  _BYTE *v32; // rax
  size_t v33; // r15
  size_t v34; // r13
  signed __int64 v35; // r12
  __int64 v36; // rax
  bool v37; // cl
  bool v38; // r13
  int v39; // [rsp+4h] [rbp-7Ch]
  size_t v40; // [rsp+10h] [rbp-70h]
  unsigned __int64 v41; // [rsp+10h] [rbp-70h]
  size_t *v42; // [rsp+18h] [rbp-68h]
  signed __int64 v43; // [rsp+20h] [rbp-60h]
  signed __int64 v44; // [rsp+28h] [rbp-58h]
  size_t v45; // [rsp+28h] [rbp-58h]
  size_t **v46; // [rsp+30h] [rbp-50h]
  __int64 v47; // [rsp+38h] [rbp-48h]
  size_t n; // [rsp+40h] [rbp-40h]
  size_t nb; // [rsp+40h] [rbp-40h]
  size_t na; // [rsp+40h] [rbp-40h]
  int v51; // [rsp+48h] [rbp-38h]
  unsigned int v52; // [rsp+4Ch] [rbp-34h]

  v4 = a1[6];
  if ( !v4 )
  {
    *a3 = 0;
    return 0;
  }
  v7 = *a2;
  v47 = *((_QWORD *)a1 + 1);
  v9 = a2[3];
  v10 = a2[2];
  v11 = v9 - v10;
  v12 = v9 - v10;
  if ( v9 == v10 )
  {
    a1 = 0;
  }
  else
  {
    if ( v11 > 0x7FFFFFFFFFFFFFF8LL )
LABEL_66:
      sub_4261EA(a1, v10, v11);
    v13 = (_DWORD *)sub_22077B0(v11);
    v14 = a2[2];
    a1 = v13;
    v11 = (_BYTE *)a2[3] - v14;
    if ( v14 == a2[3] )
    {
      if ( !v13 )
        goto LABEL_9;
    }
    else
    {
      a1 = memmove(v13, v14, v11);
    }
    j_j___libc_free_0(a1, v12);
  }
LABEL_9:
  v51 = 1;
  v39 = v4 - 1;
  v52 = (v4 - 1) & (unsigned int)v7;
  v42 = 0;
  v46 = a3;
  while ( 1 )
  {
    v15 = (size_t *)(v47 + 40LL * v52);
    v10 = (_BYTE *)v15[2];
    v16 = *v15;
    n = v15[1];
    v17 = (_BYTE *)v15[3];
    v18 = v17 - v10;
    v43 = v17 - v10;
    if ( v17 == v10 )
    {
      v20 = 0;
    }
    else
    {
      if ( v18 > 0x7FFFFFFFFFFFFFF8LL )
        goto LABEL_66;
      a1 = (_DWORD *)(v15[3] - (_QWORD)v10);
      v19 = sub_22077B0(v18);
      v10 = (_BYTE *)v15[2];
      v20 = (_DWORD *)v19;
      v17 = (_BYTE *)v15[3];
      v18 = v17 - v10;
    }
    if ( v17 != v10 )
    {
      a1 = v20;
      memmove(v20, v10, v18);
    }
    v21 = a2[3];
    v22 = a2[2];
    v10 = *a2;
    v11 = (size_t)a2[1];
    v23 = v21 - v22;
    v44 = v21 - v22;
    if ( v21 == v22 )
    {
      v25 = 0;
      a1 = 0;
    }
    else
    {
      v40 = (size_t)a2[1];
      if ( v23 > 0x7FFFFFFFFFFFFFF8LL )
        goto LABEL_66;
      v24 = sub_22077B0(v23);
      v22 = a2[2];
      v11 = v40;
      a1 = (_DWORD *)v24;
      v21 = a2[3];
      v23 = v21 - v22;
      v25 = v21 - v22;
    }
    LOBYTE(v16) = v16 == (_QWORD)v10;
    LOBYTE(v11) = n == v11;
    v26 = v11 & v16;
    LOBYTE(v11) = v18 == v23;
    v5 = v11 & v26;
    if ( v22 != v21 )
    {
      v41 = v23;
      nb = v25;
      v27 = memmove(a1, v22, v25);
      v25 = nb;
      a1 = v27;
      if ( (_BYTE)v5 && v41 )
LABEL_51:
        LOBYTE(v5) = memcmp(a1, v20, v25) == 0;
LABEL_20:
      j_j___libc_free_0(a1, v44);
      goto LABEL_21;
    }
    if ( (_BYTE)v5 && v23 )
      goto LABEL_51;
    if ( a1 )
      goto LABEL_20;
LABEL_21:
    if ( v20 )
    {
      a1 = v20;
      j_j___libc_free_0(v20, v43);
    }
    if ( (_BYTE)v5 )
    {
      *v46 = v15;
      return v5;
    }
    v10 = (_BYTE *)v15[2];
    na = *v15;
    v45 = v15[1];
    v28 = (_BYTE *)v15[3];
    v29 = v28 - v10;
    v30 = v28 - v10;
    if ( v28 == v10 )
    {
      a1 = 0;
    }
    else
    {
      if ( v29 > 0x7FFFFFFFFFFFFFF8LL )
        goto LABEL_66;
      v31 = sub_22077B0(v29);
      v10 = (_BYTE *)v15[2];
      a1 = (_DWORD *)v31;
      v28 = (_BYTE *)v15[3];
      v29 = v28 - v10;
    }
    if ( v10 != v28 )
      a1 = memmove(a1, v10, v29);
    if ( !(v29 | na) && v45 == -1 )
      break;
    if ( a1 )
      j_j___libc_free_0(a1, v30);
    v32 = (_BYTE *)v15[3];
    v10 = (_BYTE *)v15[2];
    v33 = *v15;
    v34 = v15[1];
    v11 = v32 - v10;
    v35 = v32 - v10;
    if ( v32 == v10 )
    {
      a1 = 0;
    }
    else
    {
      if ( v11 > 0x7FFFFFFFFFFFFFF8LL )
        goto LABEL_66;
      v36 = sub_22077B0(v11);
      v10 = (_BYTE *)v15[2];
      a1 = (_DWORD *)v36;
      v32 = (_BYTE *)v15[3];
      v11 = v32 - v10;
    }
    v37 = v34 == -2 && (v11 | v33 | (unsigned __int64)v42) == 0;
    v38 = v37;
    if ( v32 != v10 )
    {
      a1 = memmove(a1, v10, v11);
      if ( !v38 )
        goto LABEL_47;
LABEL_63:
      j_j___libc_free_0(a1, v35);
      v42 = (size_t *)(v47 + 40LL * v52);
      goto LABEL_48;
    }
    if ( v37 )
    {
      if ( a1 )
        goto LABEL_63;
      v42 = (size_t *)(v47 + 40LL * v52);
    }
    else if ( a1 )
    {
LABEL_47:
      j_j___libc_free_0(a1, v35);
    }
LABEL_48:
    v52 = v39 & (v51 + v52);
    ++v51;
  }
  if ( a1 )
    j_j___libc_free_0(a1, v30);
  if ( v42 )
    v15 = v42;
  *v46 = v15;
  return v5;
}
