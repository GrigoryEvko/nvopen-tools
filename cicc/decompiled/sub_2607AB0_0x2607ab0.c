// Function: sub_2607AB0
// Address: 0x2607ab0
//
char __fastcall sub_2607AB0(char *a1, __int64 *a2, __int64 a3, signed __int64 a4, __int64 a5, char *a6, __int64 a7)
{
  __int64 *v7; // r15
  char *v8; // r13
  char *v9; // r12
  __int64 v10; // rbx
  signed __int64 v11; // rax
  __int64 *v12; // r8
  char *v13; // r14
  __int64 v14; // rsi
  int v15; // r9d
  __int64 v16; // rcx
  __int64 v17; // rdx
  __int64 v18; // rax
  bool v19; // of
  signed __int64 v20; // rax
  signed __int64 v21; // rdx
  __int64 v22; // r11
  __int64 v23; // rax
  int v24; // r10d
  _QWORD *v25; // rbx
  size_t v26; // rdx
  char *v27; // r9
  char *v28; // r14
  char *v29; // r9
  __int64 v30; // rdi
  int v31; // r8d
  __int64 v32; // rcx
  __int64 v33; // rdx
  __int64 v34; // rax
  signed __int64 v35; // rax
  signed __int64 v36; // rdx
  __int64 v37; // r11
  __int64 v38; // rax
  int v39; // r10d
  __int64 *v40; // rax
  __int64 v41; // r10
  signed __int64 v42; // rcx
  char *v43; // r11
  __int64 v44; // r8
  char *v45; // r14
  __int64 *v46; // rax
  bool v47; // cc
  int src; // [rsp+10h] [rbp-60h]
  __int64 v50; // [rsp+18h] [rbp-58h]
  signed __int64 v51; // [rsp+20h] [rbp-50h]
  __int64 v52; // [rsp+20h] [rbp-50h]
  __int64 v53; // [rsp+28h] [rbp-48h]
  signed __int64 v54; // [rsp+28h] [rbp-48h]
  __int64 *v55; // [rsp+28h] [rbp-48h]
  __int64 *v56; // [rsp+30h] [rbp-40h]
  size_t v57; // [rsp+38h] [rbp-38h]
  __int64 v58; // [rsp+38h] [rbp-38h]

  while ( 1 )
  {
    v7 = (__int64 *)a3;
    v8 = a6;
    v9 = a1;
    v10 = a5;
    v11 = a7;
    if ( a5 <= a7 )
      v11 = a5;
    if ( a4 <= v11 )
    {
      v12 = a2;
      if ( a2 != (__int64 *)a1 )
      {
        LOBYTE(v11) = (unsigned __int8)memmove(a6, a1, (char *)a2 - a1);
        v12 = a2;
      }
      v13 = &v8[(char *)a2 - a1];
      if ( v13 == v8 )
        return v11;
      while ( 1 )
      {
        if ( v7 == v12 )
        {
          v26 = v13 - v8;
          goto LABEL_44;
        }
        v14 = *(_QWORD *)v8;
        v15 = 1;
        v16 = *v12;
        v17 = *(_QWORD *)(*(_QWORD *)v8 + 296LL);
        v18 = *(_QWORD *)(*(_QWORD *)v8 + 280LL);
        if ( *(_DWORD *)(*(_QWORD *)v8 + 304LL) != 1 )
          v15 = *(_DWORD *)(v14 + 288);
        v19 = __OFSUB__(v18, v17);
        v20 = v18 - v17;
        if ( v19 )
        {
          v47 = v17 <= 0;
          v21 = 0x8000000000000000LL;
          if ( v47 )
            v21 = 0x7FFFFFFFFFFFFFFFLL;
        }
        else
        {
          v21 = v20;
        }
        v22 = *(_QWORD *)(v16 + 296);
        v23 = *(_QWORD *)(v16 + 280);
        v24 = 1;
        if ( *(_DWORD *)(v16 + 304) != 1 )
          v24 = *(_DWORD *)(v16 + 288);
        v19 = __OFSUB__(v23, v22);
        v11 = v23 - v22;
        if ( v19 )
        {
          if ( v22 > 0 )
          {
            if ( v24 == v15 )
            {
LABEL_21:
              v8 += 8;
              v16 = v14;
              goto LABEL_10;
            }
            goto LABEL_20;
          }
          v11 = 0x7FFFFFFFFFFFFFFFLL;
        }
        if ( v24 == v15 )
        {
          if ( v11 <= v21 )
            goto LABEL_21;
          goto LABEL_9;
        }
LABEL_20:
        if ( v15 >= v24 )
          goto LABEL_21;
LABEL_9:
        ++v12;
LABEL_10:
        *(_QWORD *)a1 = v16;
        a1 += 8;
        if ( v13 == v8 )
          return v11;
      }
    }
    if ( a5 <= a7 )
      break;
    v51 = a4;
    if ( a4 > a5 )
    {
      v58 = a4 / 2;
      v55 = (__int64 *)&a1[8 * (a4 / 2)];
      v46 = sub_25F7EB0(a2, a3, v55);
      v43 = (char *)v55;
      v42 = v51;
      v56 = v46;
      v41 = a7;
      v44 = v46 - a2;
    }
    else
    {
      v53 = a5 / 2;
      v56 = &a2[a5 / 2];
      v40 = sub_25F7DB0((__int64 *)a1, (__int64)a2, v56);
      v41 = a7;
      v42 = v51;
      v43 = (char *)v40;
      v44 = v53;
      v58 = ((char *)v40 - a1) >> 3;
    }
    v50 = v41;
    v52 = v44;
    src = (int)v43;
    v54 = v42 - v58;
    v45 = sub_2607990(v43, (char *)a2, (char *)v56, v42 - v58, v44, v8, v41);
    sub_2607AB0((_DWORD)a1, src, (_DWORD)v45, v58, v52, (_DWORD)v8, v50);
    a6 = v8;
    a4 = v54;
    a2 = v56;
    a1 = v45;
    a7 = v50;
    a5 = v10 - v52;
    a3 = (__int64)v7;
  }
  v25 = (_QWORD *)a3;
  v26 = a3 - (_QWORD)a2;
  if ( v7 != a2 )
  {
    v57 = v26;
    LOBYTE(v11) = (unsigned __int8)memmove(a6, a2, v26);
    v26 = v57;
  }
  v27 = &v8[v26];
  if ( a2 == (__int64 *)a1 )
  {
    if ( v8 != v27 )
    {
      a1 = (char *)v7 - v26;
LABEL_44:
      LOBYTE(v11) = (unsigned __int8)memmove(a1, v8, v26);
    }
  }
  else if ( v8 != v27 )
  {
    v28 = (char *)(a2 - 1);
    v29 = v27 - 8;
    while ( 1 )
    {
      while ( 1 )
      {
        v30 = *(_QWORD *)v28;
        v31 = 1;
        v32 = *(_QWORD *)v29;
        v33 = *(_QWORD *)(*(_QWORD *)v28 + 296LL);
        v34 = *(_QWORD *)(*(_QWORD *)v28 + 280LL);
        if ( *(_DWORD *)(*(_QWORD *)v28 + 304LL) != 1 )
          v31 = *(_DWORD *)(v30 + 288);
        v19 = __OFSUB__(v34, v33);
        v35 = v34 - v33;
        if ( v19 )
        {
          v47 = v33 <= 0;
          v36 = 0x7FFFFFFFFFFFFFFFLL;
          if ( !v47 )
            v36 = 0x8000000000000000LL;
        }
        else
        {
          v36 = v35;
        }
        v37 = *(_QWORD *)(v32 + 296);
        v38 = *(_QWORD *)(v32 + 280);
        v39 = 1;
        if ( *(_DWORD *)(v32 + 304) != 1 )
          v39 = *(_DWORD *)(v32 + 288);
        v19 = __OFSUB__(v38, v37);
        v11 = v38 - v37;
        if ( v19 )
        {
          v11 = 0x7FFFFFFFFFFFFFFFLL;
          if ( v37 > 0 )
            v11 = 0x8000000000000000LL;
        }
        LOBYTE(v11) = v39 == v31 ? v11 > v36 : v31 < v39;
        --v25;
        if ( (_BYTE)v11 )
          break;
        *v25 = v32;
        if ( v8 == v29 )
          return v11;
        v29 -= 8;
      }
      *v25 = v30;
      if ( v9 == v28 )
        break;
      v28 -= 8;
    }
    if ( v8 != v29 + 8 )
    {
      v26 = v29 + 8 - v8;
      a1 = (char *)v25 - v26;
      goto LABEL_44;
    }
  }
  return v11;
}
