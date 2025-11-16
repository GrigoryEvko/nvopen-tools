// Function: sub_2CB0040
// Address: 0x2cb0040
//
__int64 __fastcall sub_2CB0040(__int64 a1, __int64 a2, _QWORD *a3, __int64 a4, __int64 a5, unsigned __int64 a6)
{
  __int64 v7; // rax
  __int64 v8; // r12
  int v9; // r13d
  bool v10; // bl
  __int64 v11; // rax
  __int64 v12; // rax
  __int64 *v14; // r15
  __int64 v16; // rsi
  unsigned __int8 *v17; // rsi
  bool v18; // zf
  __int64 *v19; // rax
  _BYTE *v20; // rsi
  __int64 *v21; // r13
  unsigned int v22; // eax
  __int64 v23; // rdx
  __int64 *v24; // r15
  __int64 *v25; // r15
  __int64 v26; // r14
  __int64 v27; // rbx
  __int64 v28; // r12
  __int64 v29; // rdi
  int v30; // eax
  unsigned int v31; // eax
  __int64 v32; // rdi
  int v33; // eax
  int v34; // eax
  __int64 *v35; // r13
  __int64 v36; // rsi
  __int64 v37; // r13
  __int64 v38; // rbx
  __int64 v39; // r12
  unsigned int v40; // eax
  __int64 v41; // rdx
  __int64 v42; // rdx
  int v43; // eax
  int v44; // eax
  int v45; // r13d
  bool v46; // bl
  __int64 v47; // rax
  unsigned __int8 *v50; // [rsp+18h] [rbp-58h]
  __int64 *v53; // [rsp+30h] [rbp-40h]
  __int64 v54; // [rsp+38h] [rbp-38h]
  unsigned __int8 *v55; // [rsp+38h] [rbp-38h]

  v7 = *(unsigned int *)(a4 + 8);
  v8 = *(_QWORD *)(a5 + 104);
  if ( !(_DWORD)v7 )
  {
    v9 = *(_DWORD *)(a5 + 112);
    v10 = *(_QWORD *)a5 == 0x200000002LL;
    v11 = sub_BCCE00(a3, 0x20u);
    v12 = sub_ACD640(v11, 0, 0);
    *(_DWORD *)(a1 + 8) = v10 + 1;
    *(_QWORD *)a1 = v12;
    *(_DWORD *)(a1 + 12) = v9;
    *(_QWORD *)(a1 + 16) = v8;
    return a1;
  }
  v14 = *(__int64 **)a4;
  v54 = *(_QWORD *)a4 + 24 * v7;
  v53 = (__int64 *)v54;
  while ( 1 )
  {
    v20 = (_BYTE *)*v14;
    if ( *(_BYTE *)*v14 <= 0x1Cu )
    {
      if ( *(_QWORD *)a5 != 0x200000002LL )
        goto LABEL_18;
      v50 = (unsigned __int8 *)*v14;
      v20 = v50;
      v32 = sub_B43CA0(*(_QWORD *)(a5 + 104));
      v33 = *v50;
      if ( (unsigned __int8)v33 <= 0x1Cu )
      {
        v20 = v50;
        if ( (unsigned int)sub_2CAFE10(v32 + 312, (__int64)v50) == 2 )
          goto LABEL_18;
      }
      else
      {
        v34 = v33 - 29;
        if ( v34 == 39 || v34 == 26 )
        {
LABEL_18:
          if ( sub_2CAFF80(a6, (__int64)v20, 0x20u, 0) )
          {
            v21 = v14;
            goto LABEL_20;
          }
          goto LABEL_15;
        }
      }
      goto LABEL_15;
    }
    if ( !(_BYTE)qword_50130E8 )
      goto LABEL_55;
    if ( v20 == (_BYTE *)v8 )
    {
      v17 = (unsigned __int8 *)v8;
      if ( *(_QWORD *)a5 != 0x200000002LL )
        goto LABEL_12;
      goto LABEL_25;
    }
    if ( !(unsigned __int8)sub_B19DB0(a2, (__int64)v20, v8) )
    {
LABEL_55:
      v16 = v14[2];
      if ( !v16 || v16 != v8 && !(unsigned __int8)sub_B19DB0(a2, v16, v8) )
        break;
    }
    v17 = (unsigned __int8 *)*v14;
    if ( *(_QWORD *)a5 != 0x200000002LL )
      goto LABEL_12;
LABEL_25:
    v29 = sub_B43CA0(*(_QWORD *)(a5 + 104));
    v30 = *v17;
    if ( (unsigned __int8)v30 <= 0x1Cu )
    {
      if ( (unsigned int)sub_2CAFE10(v29 + 312, (__int64)v17) == 2 )
      {
LABEL_12:
        v18 = !sub_2CAFF80(a6, (__int64)v17, 0x20u, 0);
        v19 = v53;
        if ( !v18 )
          v19 = v14;
        v53 = v19;
      }
LABEL_15:
      v14 += 3;
      if ( (__int64 *)v54 == v14 )
        break;
    }
    else
    {
      v31 = v30 - 29;
      if ( v31 == 39 )
        goto LABEL_12;
      if ( v31 > 0x27 )
        goto LABEL_15;
      if ( v31 == 26 )
        goto LABEL_12;
      v14 += 3;
      if ( (__int64 *)v54 == v14 )
        break;
    }
  }
  v21 = v53;
LABEL_20:
  v22 = *(_DWORD *)(a4 + 8);
  v23 = 3LL * v22;
  v24 = *(__int64 **)a4;
  if ( v21 != (__int64 *)v54 )
  {
    v25 = &v24[v23];
    v26 = *v21;
    v27 = v21[2];
    v28 = v21[1];
    if ( v25 != v21 + 3 )
    {
      memmove(v21, v21 + 3, (char *)v25 - (char *)(v21 + 3));
      v22 = *(_DWORD *)(a4 + 8);
    }
    *(_DWORD *)(a4 + 8) = v22 - 1;
    *(_QWORD *)a1 = v26;
    *(_QWORD *)(a1 + 8) = v28;
    *(_QWORD *)(a1 + 16) = v27;
    return a1;
  }
  v35 = &v24[v23];
  if ( &v24[v23] == v24 )
  {
LABEL_47:
    v45 = *(_DWORD *)(a5 + 112);
    v46 = *(_QWORD *)a5 == 0x200000002LL;
    v47 = sub_BCCE00(a3, 0x20u);
    *(_QWORD *)a1 = sub_ACD640(v47, 0, 0);
    *(_DWORD *)(a1 + 8) = v46 + 1;
    *(_DWORD *)(a1 + 12) = v45;
    *(_QWORD *)(a1 + 16) = v8;
    return a1;
  }
  while ( 1 )
  {
LABEL_38:
    v36 = *v24;
    if ( *(_QWORD *)a5 != 0x200000002LL )
      goto LABEL_39;
    v55 = (unsigned __int8 *)*v24;
    v36 = (__int64)v55;
    v42 = sub_B43CA0(*(_QWORD *)(a5 + 104));
    v43 = *v55;
    if ( (unsigned __int8)v43 > 0x1Cu )
      break;
    v36 = (__int64)v55;
    if ( (unsigned int)sub_2CAFE10(v42 + 312, (__int64)v55) == 2 )
      goto LABEL_39;
    v24 += 3;
    if ( v35 == v24 )
      goto LABEL_47;
  }
  v44 = v43 - 29;
  if ( v44 != 39 && v44 != 26 )
  {
LABEL_46:
    v24 += 3;
    if ( v35 == v24 )
      goto LABEL_47;
    goto LABEL_38;
  }
LABEL_39:
  if ( !sub_2CAFF80(a6, v36, 0x20u, 0) )
    goto LABEL_46;
  v37 = *v24;
  v38 = v24[2];
  v39 = v24[1];
  v40 = *(_DWORD *)(a4 + 8);
  v41 = *(_QWORD *)a4 + 24LL * v40;
  if ( (__int64 *)v41 != v24 + 3 )
  {
    memmove(v24, v24 + 3, v41 - (_QWORD)(v24 + 3));
    v40 = *(_DWORD *)(a4 + 8);
  }
  *(_DWORD *)(a4 + 8) = v40 - 1;
  *(_QWORD *)a1 = v37;
  *(_QWORD *)(a1 + 8) = v39;
  *(_QWORD *)(a1 + 16) = v38;
  return a1;
}
