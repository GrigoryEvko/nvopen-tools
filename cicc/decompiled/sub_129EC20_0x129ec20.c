// Function: sub_129EC20
// Address: 0x129ec20
//
_QWORD *__fastcall sub_129EC20(_QWORD *a1, __int64 a2, __int64 a3, __int64 a4)
{
  _QWORD *v4; // r14
  const char *v7; // r15
  size_t v8; // rax
  __int64 v9; // r15
  bool v10; // al
  _QWORD *v11; // rdx
  unsigned __int64 v12; // r9
  unsigned __int64 v13; // rcx
  _BYTE *v14; // rdx
  __int64 v15; // rdx
  __int64 v16; // rcx
  __int64 v17; // r8
  __int64 v18; // r9
  bool v19; // al
  _QWORD *v20; // rdi
  __int64 v21; // rax
  _QWORD *v22; // rdi
  __int64 v23; // rax
  const char *v24; // r13
  size_t v25; // rdx
  __int64 v26; // rcx
  __int64 v27; // rbx
  _QWORD *v28; // rax
  unsigned __int64 v29; // r13
  unsigned __int64 v30; // rdx
  _BYTE *v31; // rax
  unsigned __int64 v33; // rcx
  __int64 v34; // r9
  _BYTE *v35; // rdx
  unsigned __int64 v36; // r13
  unsigned __int64 v37; // rdx
  _BYTE *v38; // rax
  bool v39; // [rsp+Fh] [rbp-31h]
  bool v40; // [rsp+Fh] [rbp-31h]

  v4 = a1 + 2;
  *a1 = a1 + 2;
  a1[1] = 0;
  *((_BYTE *)a1 + 16) = 0;
  if ( (*(_BYTE *)(a3 + 89) & 8) == 0 || (v7 = *(const char **)(a3 + 24)) == 0 )
  {
    v7 = *(const char **)(a3 + 8);
    if ( !v7 )
      return a1;
  }
  v8 = strlen(v7);
  sub_2241130(a1, 0, 0, v7, v8);
  v9 = a1[1];
  v10 = v9 != 0 && a4 != 0;
  if ( !v10 )
    return a1;
  v11 = (_QWORD *)*a1;
  if ( *(_BYTE *)(*a1 + v9 - 1) == 60 )
  {
    if ( v4 == v11 )
      v33 = 15;
    else
      v33 = a1[2];
    v34 = v9 + 1;
    if ( v33 < v9 + 1 )
    {
      sub_2240BB0(a1, a1[1], 0, 0, 1);
      v11 = (_QWORD *)*a1;
      v34 = v9 + 1;
      v10 = v9 != 0 && a4 != 0;
    }
    *((_BYTE *)v11 + v9) = 32;
    v35 = (_BYTE *)*a1;
    a1[1] = v34;
    v35[v34] = 0;
    v9 = a1[1];
    v11 = (_QWORD *)*a1;
  }
  v12 = v9 + 1;
  if ( v4 == v11 )
    v13 = 15;
  else
    v13 = a1[2];
  if ( v12 > v13 )
  {
    v40 = v10;
    sub_2240BB0(a1, v9, 0, 0, 1);
    v11 = (_QWORD *)*a1;
    v12 = v9 + 1;
    v10 = v40;
  }
  *((_BYTE *)v11 + v9) = 60;
  v14 = (_BYTE *)*a1;
  a1[1] = v12;
  v14[v12] = 0;
  v39 = v10;
  sub_823800(*(_QWORD *)(a2 + 640));
  v19 = v39;
  while ( *(_BYTE *)(a4 + 8) == 3 )
  {
LABEL_18:
    a4 = *(_QWORD *)a4;
    if ( !a4 )
      goto LABEL_19;
  }
  if ( !v19 )
    goto LABEL_14;
  while ( 1 )
  {
    sub_747370(a4, a2 + 648);
    a4 = *(_QWORD *)a4;
    if ( !a4 )
      break;
    if ( *(_BYTE *)(a4 + 8) == 3 )
    {
      v19 = 0;
      goto LABEL_18;
    }
LABEL_14:
    v20 = *(_QWORD **)(a2 + 640);
    v21 = v20[2];
    if ( (unsigned __int64)(v21 + 1) > v20[1] )
    {
      sub_823810(v20, v21 + 1, v15, v16, v17, v18);
      v20 = *(_QWORD **)(a2 + 640);
      v21 = v20[2];
    }
    *(_BYTE *)(v20[4] + v21) = 44;
    ++*(_QWORD *)(*(_QWORD *)(a2 + 640) + 16LL);
  }
LABEL_19:
  v22 = *(_QWORD **)(a2 + 640);
  v23 = v22[2];
  if ( (unsigned __int64)(v23 + 1) > v22[1] )
  {
    sub_823810(v22, v23 + 1, v15, v16, v17, v18);
    v22 = *(_QWORD **)(a2 + 640);
    v23 = v22[2];
  }
  *(_BYTE *)(v22[4] + v23) = 0;
  ++*(_QWORD *)(*(_QWORD *)(a2 + 640) + 16LL);
  v24 = *(const char **)(*(_QWORD *)(a2 + 640) + 32LL);
  v25 = strlen(v24);
  if ( v25 > 0x3FFFFFFFFFFFFFFFLL - a1[1] )
    sub_4262D8((__int64)"basic_string::append");
  sub_2241490(a1, v24, v25, v26);
  v27 = a1[1];
  v28 = (_QWORD *)*a1;
  if ( *(_BYTE *)(*a1 + v27 - 1) == 62 )
  {
    v36 = v27 + 1;
    if ( v4 == v28 )
      v37 = 15;
    else
      v37 = a1[2];
    if ( v36 > v37 )
    {
      sub_2240BB0(a1, a1[1], 0, 0, 1);
      v28 = (_QWORD *)*a1;
    }
    *((_BYTE *)v28 + v27) = 32;
    v38 = (_BYTE *)*a1;
    a1[1] = v36;
    v38[v36] = 0;
    v27 = a1[1];
    v28 = (_QWORD *)*a1;
  }
  v29 = v27 + 1;
  if ( v4 == v28 )
    v30 = 15;
  else
    v30 = a1[2];
  if ( v29 > v30 )
  {
    sub_2240BB0(a1, v27, 0, 0, 1);
    v28 = (_QWORD *)*a1;
  }
  *((_BYTE *)v28 + v27) = 62;
  v31 = (_BYTE *)*a1;
  a1[1] = v29;
  v31[v29] = 0;
  return a1;
}
