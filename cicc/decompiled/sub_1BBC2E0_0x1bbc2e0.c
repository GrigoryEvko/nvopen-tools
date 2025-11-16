// Function: sub_1BBC2E0
// Address: 0x1bbc2e0
//
char __fastcall sub_1BBC2E0(char *a1, char *a2, __int64 a3, __int64 a4, __int64 a5, char *a6, char *a7, __int64 a8)
{
  char *v8; // r13
  char *v9; // r12
  char *v10; // rbx
  char *v11; // rax
  __int64 v12; // r14
  __int64 v13; // rbx
  char *v14; // r11
  __int64 *v15; // r9
  __int64 *v16; // rax
  int v17; // r11d
  __int64 v18; // r8
  char *v19; // r9
  __int64 *v20; // r13
  __int64 v21; // r12
  char *v22; // r10
  size_t v23; // r10
  signed __int64 v24; // r14
  char *v25; // rcx
  __int64 v26; // r15
  __int64 v27; // r14
  __int64 v28; // rdi
  unsigned int v29; // eax
  __int64 v30; // r14
  char *v31; // rsi
  char *v32; // rdi
  size_t v33; // rdx
  __int64 v34; // rcx
  char *v35; // rsi
  char *v36; // rbx
  char *v37; // r12
  __int64 v38; // r15
  __int64 v39; // r14
  __int64 v40; // rdi
  unsigned int v41; // eax
  __int64 *v42; // rax
  size_t v43; // rcx
  char *v44; // rax
  char *v45; // rax
  int v47; // [rsp+0h] [rbp-70h]
  char *v48; // [rsp+0h] [rbp-70h]
  __int64 v49; // [rsp+8h] [rbp-68h]
  int v50; // [rsp+8h] [rbp-68h]
  int v51; // [rsp+8h] [rbp-68h]
  int v52; // [rsp+8h] [rbp-68h]
  int v53; // [rsp+8h] [rbp-68h]
  char *src; // [rsp+18h] [rbp-58h]
  void *srca; // [rsp+18h] [rbp-58h]
  void *srcb; // [rsp+18h] [rbp-58h]
  void *srcc; // [rsp+18h] [rbp-58h]
  __int64 srcd; // [rsp+18h] [rbp-58h]
  int srce; // [rsp+18h] [rbp-58h]
  void *srcf; // [rsp+18h] [rbp-58h]
  void *srcg; // [rsp+18h] [rbp-58h]
  int srch; // [rsp+18h] [rbp-58h]
  int srci; // [rsp+18h] [rbp-58h]
  char *v65; // [rsp+20h] [rbp-50h]
  char *v66; // [rsp+20h] [rbp-50h]
  void *v67; // [rsp+20h] [rbp-50h]
  void *v68; // [rsp+20h] [rbp-50h]
  char *v69; // [rsp+20h] [rbp-50h]
  char *v70; // [rsp+20h] [rbp-50h]
  void *v71; // [rsp+20h] [rbp-50h]
  void *v72; // [rsp+20h] [rbp-50h]
  void *v73; // [rsp+20h] [rbp-50h]
  void *v74; // [rsp+20h] [rbp-50h]
  char *dest; // [rsp+28h] [rbp-48h]
  char *desta; // [rsp+28h] [rbp-48h]
  char *destb; // [rsp+28h] [rbp-48h]
  __int64 *v78; // [rsp+30h] [rbp-40h]
  __int64 v79; // [rsp+30h] [rbp-40h]
  char *v80; // [rsp+30h] [rbp-40h]

  v8 = a1;
  v9 = a2;
  v10 = (char *)a3;
  v11 = (char *)a5;
  if ( (__int64)a7 <= a5 )
    v11 = a7;
  if ( a4 > (__int64)v11 )
  {
    v12 = a5;
    if ( (__int64)a7 >= a5 )
      goto LABEL_48;
    v13 = a4;
    v14 = a1;
    dest = a6;
    v15 = (__int64 *)a2;
    while ( 1 )
    {
      if ( v12 < v13 )
      {
        srce = (int)v14;
        v70 = (char *)v15;
        v21 = v13 / 2;
        v20 = (__int64 *)&v14[8 * (v13 / 2)];
        v42 = sub_1BBA9A0(v15, a3, v20, a8);
        v19 = v70;
        v17 = srce;
        v78 = v42;
        v18 = ((char *)v42 - v70) >> 3;
      }
      else
      {
        src = (char *)v15;
        v65 = v14;
        v78 = &v15[v12 / 2];
        v16 = sub_1BBAAE0(v14, (__int64)v15, v78, a8);
        v17 = (int)v65;
        v18 = v12 / 2;
        v19 = src;
        v20 = v16;
        v21 = ((char *)v16 - v65) >> 3;
      }
      v13 -= v21;
      if ( v13 <= v18 || v18 > (__int64)a7 )
      {
        if ( v13 > (__int64)a7 )
        {
          srci = v17;
          v74 = (void *)v18;
          v45 = sub_1BBB7C0((char *)v20, v19, (char *)v78);
          v17 = srci;
          v18 = (__int64)v74;
          v22 = v45;
        }
        else
        {
          v22 = (char *)v78;
          if ( v13 )
          {
            v43 = v19 - (char *)v20;
            if ( v19 != (char *)v20 )
            {
              v48 = v19;
              v52 = v17;
              srcf = (void *)v18;
              v71 = (void *)(v19 - (char *)v20);
              memmove(dest, v20, v19 - (char *)v20);
              v19 = v48;
              v17 = v52;
              v18 = (__int64)srcf;
              v43 = (size_t)v71;
            }
            if ( v19 != (char *)v78 )
            {
              v53 = v17;
              srcg = (void *)v43;
              v72 = (void *)v18;
              memmove(v20, v19, (char *)v78 - v19);
              v17 = v53;
              v43 = (size_t)srcg;
              v18 = (__int64)v72;
            }
            v22 = (char *)v78 - v43;
            if ( v43 )
            {
              srch = v17;
              v73 = (void *)v18;
              v44 = (char *)memmove((char *)v78 - v43, dest, v43);
              v18 = (__int64)v73;
              v17 = srch;
              v22 = v44;
            }
          }
        }
      }
      else
      {
        v22 = (char *)v20;
        if ( v18 )
        {
          v23 = (char *)v78 - v19;
          if ( v19 != (char *)v78 )
          {
            v47 = v17;
            v49 = v18;
            srca = (void *)((char *)v78 - v19);
            v66 = v19;
            memmove(dest, v19, (char *)v78 - v19);
            v17 = v47;
            v18 = v49;
            v23 = (size_t)srca;
            v19 = v66;
          }
          if ( v19 != (char *)v20 )
          {
            v50 = v17;
            srcb = (void *)v23;
            v67 = (void *)v18;
            memmove((char *)v78 - (v19 - (char *)v20), v20, v19 - (char *)v20);
            v17 = v50;
            v23 = (size_t)srcb;
            v18 = (__int64)v67;
          }
          if ( v23 )
          {
            v51 = v17;
            srcc = (void *)v18;
            v68 = (void *)v23;
            memmove(v20, dest, v23);
            v17 = v51;
            v18 = (__int64)srcc;
            v23 = (size_t)v68;
          }
          v22 = (char *)v20 + v23;
        }
      }
      srcd = v18;
      v69 = v22;
      sub_1BBC2E0(v17, (_DWORD)v20, (_DWORD)v22, v21, v18, (_DWORD)dest, (__int64)a7, a8);
      v11 = a7;
      v12 -= srcd;
      if ( v12 <= (__int64)a7 )
        v11 = (char *)v12;
      if ( v13 <= (__int64)v11 )
        break;
      if ( v12 <= (__int64)a7 )
      {
        v10 = (char *)a3;
        a6 = dest;
        v8 = v69;
        v9 = (char *)v78;
LABEL_48:
        if ( v10 != v9 )
          a6 = (char *)memmove(a6, v9, v10 - v9);
        v11 = &a6[v10 - v9];
        if ( v8 == v9 )
        {
          if ( a6 == v11 )
            return (char)v11;
          v33 = v10 - v9;
          v32 = v9;
          goto LABEL_89;
        }
        if ( a6 == v11 )
          return (char)v11;
        v34 = a8;
        v35 = v9 - 8;
        v36 = v10 - 8;
        v37 = v11 - 8;
        while ( 2 )
        {
          v38 = *(_QWORD *)v37;
          v39 = *(_QWORD *)v35;
          LOBYTE(v11) = *(_QWORD *)v35 == 0;
          if ( (unsigned __int8)v11 | (*(_QWORD *)v37 == 0) || v39 == v38 )
            goto LABEL_60;
          if ( v38 == *(_QWORD *)(v39 + 8) )
            goto LABEL_69;
          if ( v39 == *(_QWORD *)(v38 + 8)
            || (LODWORD(v11) = *(_DWORD *)(v39 + 16), *(_DWORD *)(v38 + 16) >= (unsigned int)v11) )
          {
LABEL_60:
            *(_QWORD *)v36 = v38;
            if ( a6 == v37 )
              return (char)v11;
            v37 -= 8;
            goto LABEL_62;
          }
          v40 = *(_QWORD *)(v34 + 1352);
          if ( *(_BYTE *)(v40 + 72) )
          {
            LODWORD(v11) = *(_DWORD *)(v38 + 48);
            if ( *(_DWORD *)(v39 + 48) < (unsigned int)v11 )
              goto LABEL_60;
            LODWORD(v11) = *(_DWORD *)(v38 + 52);
            if ( *(_DWORD *)(v39 + 52) > (unsigned int)v11 )
              goto LABEL_60;
LABEL_69:
            *(_QWORD *)v36 = v39;
            if ( v35 == v8 )
            {
              if ( a6 == v37 + 8 )
                return (char)v11;
              v33 = v37 + 8 - a6;
              v32 = &v36[-v33];
LABEL_89:
              v31 = a6;
LABEL_45:
              LOBYTE(v11) = (unsigned __int8)memmove(v32, v31, v33);
              return (char)v11;
            }
            v35 -= 8;
LABEL_62:
            v36 -= 8;
            continue;
          }
          break;
        }
        v41 = *(_DWORD *)(v40 + 76) + 1;
        *(_DWORD *)(v40 + 76) = v41;
        if ( v41 > 0x20 )
        {
          desta = a6;
          v79 = v34;
          sub_15CC640(v40);
          LODWORD(v11) = *(_DWORD *)(v38 + 48);
          v34 = v79;
          a6 = desta;
          if ( *(_DWORD *)(v39 + 48) >= (unsigned int)v11 )
          {
            LODWORD(v11) = *(_DWORD *)(v38 + 52);
            if ( *(_DWORD *)(v39 + 52) <= (unsigned int)v11 )
            {
LABEL_68:
              v39 = *(_QWORD *)v35;
              goto LABEL_69;
            }
          }
        }
        else
        {
          do
          {
            v11 = (char *)v39;
            v39 = *(_QWORD *)(v39 + 8);
          }
          while ( v39 && *(_DWORD *)(v38 + 16) <= *(_DWORD *)(v39 + 16) );
          if ( (char *)v38 == v11 )
            goto LABEL_68;
        }
        v38 = *(_QWORD *)v37;
        goto LABEL_60;
      }
      v15 = v78;
      v14 = v69;
    }
    v10 = (char *)a3;
    a6 = dest;
    v8 = v69;
    v9 = (char *)v78;
  }
  v24 = v9 - v8;
  if ( v9 != v8 )
  {
    v11 = (char *)memmove(a6, v8, v9 - v8);
    a6 = v11;
  }
  v25 = &a6[v24];
  if ( a6 == &a6[v24] )
  {
LABEL_43:
    if ( v25 == a6 )
      return (char)v11;
    v31 = a6;
    v32 = v8;
    v33 = v25 - a6;
    goto LABEL_45;
  }
  do
  {
    if ( v10 == v9 )
      goto LABEL_43;
    while ( 1 )
    {
      v26 = *(_QWORD *)v9;
      v27 = *(_QWORD *)a6;
      LOBYTE(v11) = *(_QWORD *)a6 == 0;
      if ( (unsigned __int8)v11 | (*(_QWORD *)v9 == 0) || v27 == v26 )
        goto LABEL_41;
      if ( v26 == *(_QWORD *)(v27 + 8) )
        goto LABEL_72;
      if ( v27 == *(_QWORD *)(v26 + 8) )
        goto LABEL_41;
      LODWORD(v11) = *(_DWORD *)(v27 + 16);
      if ( *(_DWORD *)(v26 + 16) >= (unsigned int)v11 )
        goto LABEL_41;
      v28 = *(_QWORD *)(a8 + 1352);
      if ( *(_BYTE *)(v28 + 72) )
        break;
      v29 = *(_DWORD *)(v28 + 76) + 1;
      *(_DWORD *)(v28 + 76) = v29;
      if ( v29 > 0x20 )
      {
        destb = a6;
        v80 = v25;
        sub_15CC640(v28);
        LODWORD(v11) = *(_DWORD *)(v26 + 48);
        v25 = v80;
        a6 = destb;
        if ( *(_DWORD *)(v27 + 48) >= (unsigned int)v11 )
        {
          LODWORD(v11) = *(_DWORD *)(v26 + 52);
          if ( *(_DWORD *)(v27 + 52) <= (unsigned int)v11 )
            goto LABEL_37;
        }
      }
      else
      {
        do
        {
          v11 = (char *)v27;
          v27 = *(_QWORD *)(v27 + 8);
        }
        while ( v27 && *(_DWORD *)(v26 + 16) <= *(_DWORD *)(v27 + 16) );
        if ( (char *)v26 == v11 )
        {
LABEL_37:
          v30 = *(_QWORD *)v9;
          goto LABEL_38;
        }
      }
      v27 = *(_QWORD *)a6;
LABEL_41:
      a6 += 8;
      *(_QWORD *)v8 = v27;
      v8 += 8;
      if ( v25 == a6 )
        return (char)v11;
      if ( v10 == v9 )
        goto LABEL_43;
    }
    LODWORD(v11) = *(_DWORD *)(v26 + 48);
    if ( *(_DWORD *)(v27 + 48) < (unsigned int)v11 )
      goto LABEL_41;
    LODWORD(v11) = *(_DWORD *)(v26 + 52);
    if ( *(_DWORD *)(v27 + 52) > (unsigned int)v11 )
      goto LABEL_41;
LABEL_72:
    v30 = *(_QWORD *)v9;
LABEL_38:
    *(_QWORD *)v8 = v30;
    v9 += 8;
    v8 += 8;
  }
  while ( v25 != a6 );
  return (char)v11;
}
