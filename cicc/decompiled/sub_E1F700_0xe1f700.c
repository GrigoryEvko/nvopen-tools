// Function: sub_E1F700
// Address: 0xe1f700
//
__int64 __fastcall sub_E1F700(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  _BYTE *v6; // rax
  __int64 v7; // rdx
  __int64 result; // rax
  _BYTE *v9; // rax
  char v11; // bl
  __int64 v12; // rcx
  __int64 v13; // r15
  __int64 v14; // rdx
  __int64 v15; // rcx
  __int64 v16; // r8
  __int64 v17; // r9
  char v18; // al
  __int64 v19; // r12
  __int64 v20; // rax
  char *v21; // rdx
  __int64 v22; // r12
  char *v23; // rdi
  __int64 v24; // r12
  char v25; // al
  char *v26; // rax
  char *v27; // rax
  __int64 *v28; // r12
  __int64 v29; // rdx
  __int64 v30; // rcx
  __int64 v31; // r8
  __int64 v32; // r9
  char v33; // bl
  __int64 v34; // rax
  __int64 v35; // r13
  _BYTE *v36; // rax
  void *v37; // r12
  __int64 v38; // rdx
  __int64 v39; // rbx
  __int64 v40; // rcx
  __int64 v41; // r8
  __int64 v42; // r9
  char v43; // dl
  _QWORD *v44; // rdi
  _BYTE *v45; // r10
  __int64 v46; // rax
  size_t v47; // rdx
  char *v48; // rdi
  signed __int64 v49; // rsi
  signed __int64 v50; // rax
  char *v51; // rax
  __int64 v52; // rax
  __int64 v53; // rax
  size_t v54; // [rsp-70h] [rbp-70h]
  signed __int64 v55; // [rsp-68h] [rbp-68h]
  __int64 *v56; // [rsp-68h] [rbp-68h]
  __int64 v57; // [rsp-60h] [rbp-60h]
  __int64 v58; // [rsp-60h] [rbp-60h]
  __int64 v59; // [rsp-58h] [rbp-58h]
  __int64 v60; // [rsp-58h] [rbp-58h]
  __int64 v61; // [rsp-58h] [rbp-58h]
  __int64 v62; // [rsp-50h] [rbp-50h]
  __int64 v63[8]; // [rsp-40h] [rbp-40h] BYREF

  v6 = *(_BYTE **)a1;
  v7 = *(_QWORD *)(a1 + 8);
  if ( v7 == *(_QWORD *)a1 || *v6 != 73 )
    return 0;
  v9 = v6 + 1;
  v11 = a2;
  *(_QWORD *)a1 = v9;
  if ( !(_BYTE)a2 )
    goto LABEL_5;
  v44 = *(_QWORD **)(a1 + 664);
  *(_QWORD *)(a1 + 672) = v44;
  if ( v44 == *(_QWORD **)(a1 + 680) )
  {
    if ( v44 == (_QWORD *)(a1 + 688) )
    {
      v53 = malloc(0, a2, v7, a4, a5, a6);
      v44 = (_QWORD *)v53;
      if ( v53 )
      {
        *(_QWORD *)(a1 + 664) = v53;
        goto LABEL_82;
      }
    }
    else
    {
      a2 = 0;
      v52 = realloc(v44);
      *(_QWORD *)(a1 + 664) = v52;
      v44 = (_QWORD *)v52;
      if ( v52 )
      {
LABEL_82:
        *(_QWORD *)(a1 + 680) = v44;
        goto LABEL_53;
      }
    }
LABEL_109:
    abort();
  }
LABEL_53:
  *(_QWORD *)(a1 + 672) = v44 + 1;
  *v44 = a1 + 576;
  v7 = *(_QWORD *)(a1 + 8);
  *(_QWORD *)(a1 + 584) = *(_QWORD *)(a1 + 576);
  v9 = *(_BYTE **)a1;
LABEL_5:
  v12 = *(_QWORD *)(a1 + 24);
  v13 = *(_QWORD *)(a1 + 16);
  v62 = v12;
  do
  {
    if ( (_BYTE *)v7 != v9 && *v9 == 69 )
    {
      v35 = 0;
      *(_QWORD *)a1 = v9 + 1;
      goto LABEL_47;
    }
    if ( !v11 )
    {
      result = sub_E1F480(a1, a2, v7, v12, a5, a6);
      v63[0] = result;
      if ( !result )
        return result;
      a2 = (__int64)v63;
      sub_E18380(a1 + 16, v63, v29, v30, v31, v32);
      goto LABEL_40;
    }
    v63[0] = sub_E1F480(a1, a2, v7, v12, a5, a6);
    if ( !v63[0] )
      return 0;
    a2 = (__int64)v63;
    sub_E18380(a1 + 16, v63, v14, v15, v16, v17);
    a5 = v63[0];
    v18 = *(_BYTE *)(v63[0] + 8);
    if ( v18 == 34 )
    {
      a5 = *(_QWORD *)(v63[0] + 24);
      goto LABEL_36;
    }
    if ( v18 != 41 )
      goto LABEL_36;
    v19 = *(_QWORD *)(v63[0] + 24);
    v59 = *(_QWORD *)(v63[0] + 16);
    v20 = sub_E0E790(a1 + 816, 32, v59, v12, v63[0], a6);
    a5 = v20;
    if ( !v20 )
      return 0;
    v21 = (char *)v59;
    *(_QWORD *)(v20 + 24) = v19;
    v22 = 8 * v19;
    v12 = 4294934568LL;
    v23 = (char *)(v59 + v22);
    v24 = v22 >> 5;
    *(_QWORD *)(v20 + 16) = v59;
    *(_QWORD *)v20 = &unk_49DFCC8;
    v25 = *(_BYTE *)(v20 + 10);
    *(_WORD *)(a5 + 8) = -32728;
    *(_BYTE *)(a5 + 10) = v25 & 0xF0 | 0xA;
    if ( v24 > 0 )
    {
      v26 = (char *)v59;
      a6 = v59 + 32 * v24;
      while ( (*(_BYTE *)(*(_QWORD *)v26 + 10LL) & 3) == 1 )
      {
        if ( (*(_BYTE *)(*((_QWORD *)v26 + 1) + 10LL) & 3) != 1 )
        {
          v26 += 8;
          break;
        }
        if ( (*(_BYTE *)(*((_QWORD *)v26 + 2) + 10LL) & 3) != 1 )
        {
          v26 += 16;
          break;
        }
        if ( (*(_BYTE *)(*((_QWORD *)v26 + 3) + 10LL) & 3) != 1 )
        {
          v26 += 24;
          break;
        }
        v26 += 32;
        if ( (char *)a6 == v26 )
          goto LABEL_58;
      }
      if ( v23 != v26 )
      {
LABEL_21:
        v27 = (char *)v59;
        a6 = v24;
        while ( 1 )
        {
          a2 = (*(_BYTE *)(*(_QWORD *)v27 + 10LL) >> 2) & 3;
          if ( (_BYTE)a2 != 1 )
            break;
          a2 = (*(_BYTE *)(*((_QWORD *)v27 + 1) + 10LL) >> 2) & 3;
          if ( (_BYTE)a2 != 1 )
          {
            v27 += 8;
            break;
          }
          a2 = (*(_BYTE *)(*((_QWORD *)v27 + 2) + 10LL) >> 2) & 3;
          if ( (_BYTE)a2 != 1 )
          {
            v27 += 16;
            break;
          }
          a2 = (*(_BYTE *)(*((_QWORD *)v27 + 3) + 10LL) >> 2) & 3;
          if ( (_BYTE)a2 != 1 )
          {
            v27 += 24;
            break;
          }
          v27 += 32;
          if ( !--a6 )
            goto LABEL_66;
        }
        if ( v27 != v23 )
          goto LABEL_33;
LABEL_69:
        *(_BYTE *)(a5 + 10) = *(_BYTE *)(a5 + 10) & 0xF3 | 4;
        goto LABEL_70;
      }
      goto LABEL_63;
    }
    v26 = (char *)v59;
LABEL_58:
    v49 = v23 - v26;
    if ( v23 - v26 != 16 )
    {
      if ( v49 != 24 )
      {
        if ( v49 != 8 )
          goto LABEL_63;
        goto LABEL_61;
      }
      if ( (*(_BYTE *)(*(_QWORD *)v26 + 10LL) & 3) != 1 )
        goto LABEL_62;
      v26 += 8;
    }
    if ( (*(_BYTE *)(*(_QWORD *)v26 + 10LL) & 3) != 1 )
    {
LABEL_62:
      if ( v23 != v26 )
        goto LABEL_64;
      goto LABEL_63;
    }
    v26 += 8;
LABEL_61:
    if ( (*(_BYTE *)(*(_QWORD *)v26 + 10LL) & 3) != 1 )
      goto LABEL_62;
LABEL_63:
    *(_BYTE *)(a5 + 10) = *(_BYTE *)(a5 + 10) & 0xFC | 1;
LABEL_64:
    if ( v24 > 0 )
      goto LABEL_21;
    v27 = (char *)v59;
LABEL_66:
    a2 = v23 - v27;
    if ( v23 - v27 != 16 )
    {
      if ( a2 != 24 )
      {
        if ( a2 != 8 )
          goto LABEL_69;
        goto LABEL_96;
      }
      a2 = (*(_BYTE *)(*(_QWORD *)v27 + 10LL) >> 2) & 3;
      if ( (_BYTE)a2 != 1 )
        goto LABEL_97;
      v27 += 8;
    }
    a2 = (*(_BYTE *)(*(_QWORD *)v27 + 10LL) >> 2) & 3;
    if ( (_BYTE)a2 != 1 )
      goto LABEL_97;
    v27 += 8;
LABEL_96:
    a2 = (*(_BYTE *)(*(_QWORD *)v27 + 10LL) >> 2) & 3;
    if ( (_BYTE)a2 == 1 )
      goto LABEL_69;
LABEL_97:
    if ( v23 == v27 )
      goto LABEL_69;
LABEL_70:
    if ( v24 > 0 )
    {
LABEL_33:
      while ( *(_BYTE *)(*(_QWORD *)v21 + 9LL) >> 6 == 1 )
      {
        if ( *(_BYTE *)(*((_QWORD *)v21 + 1) + 9LL) >> 6 != 1 )
        {
          v21 += 8;
          goto LABEL_34;
        }
        if ( *(_BYTE *)(*((_QWORD *)v21 + 2) + 9LL) >> 6 != 1 )
        {
          v21 += 16;
          goto LABEL_34;
        }
        if ( *(_BYTE *)(*((_QWORD *)v21 + 3) + 9LL) >> 6 != 1 )
        {
          v21 += 24;
          goto LABEL_34;
        }
        v21 += 32;
        if ( !--v24 )
          goto LABEL_71;
      }
      goto LABEL_34;
    }
LABEL_71:
    v50 = v23 - v21;
    if ( v23 - v21 == 16 )
      goto LABEL_101;
    if ( v50 != 24 )
    {
      if ( v50 != 8 )
        goto LABEL_35;
LABEL_74:
      if ( *(_BYTE *)(*(_QWORD *)v21 + 9LL) >> 6 == 1 )
      {
LABEL_35:
        *(_BYTE *)(a5 + 9) = *(_BYTE *)(a5 + 9) & 0x3F | 0x40;
        goto LABEL_36;
      }
      goto LABEL_34;
    }
    if ( *(_BYTE *)(*(_QWORD *)v21 + 9LL) >> 6 == 1 )
    {
      v21 += 8;
LABEL_101:
      if ( *(_BYTE *)(*(_QWORD *)v21 + 9LL) >> 6 == 1 )
      {
        v21 += 8;
        goto LABEL_74;
      }
    }
LABEL_34:
    if ( v23 == v21 )
      goto LABEL_35;
LABEL_36:
    v28 = *(__int64 **)(a1 + 584);
    if ( v28 == *(__int64 **)(a1 + 592) )
    {
      v45 = *(_BYTE **)(a1 + 576);
      if ( v45 == (_BYTE *)(a1 + 600) )
      {
        v61 = 16 * (((char *)v28 - v45) >> 3);
        v54 = (char *)v28 - v45;
        v56 = *(__int64 **)(a1 + 576);
        v58 = a5;
        v48 = (char *)malloc(v61, a2, (char *)v28 - v45, v12, a5, a6);
        if ( !v48 )
          goto LABEL_109;
        a6 = v61;
        a5 = v58;
        v47 = v54;
        if ( v28 != v56 )
        {
          a2 = (__int64)v56;
          v51 = (char *)memmove(v48, v56, v54);
          a6 = v61;
          a5 = v58;
          v47 = v54;
          v48 = v51;
        }
        *(_QWORD *)(a1 + 576) = v48;
      }
      else
      {
        v60 = 16 * (((char *)v28 - v45) >> 3);
        a2 = v60;
        v55 = (char *)v28 - v45;
        v57 = a5;
        v46 = realloc(v45);
        a6 = v60;
        a5 = v57;
        *(_QWORD *)(a1 + 576) = v46;
        v47 = v55;
        v48 = (char *)v46;
        if ( !v46 )
          goto LABEL_109;
      }
      v28 = (__int64 *)&v48[v47];
      *(_QWORD *)(a1 + 592) = &v48[a6];
    }
    *(_QWORD *)(a1 + 584) = v28 + 1;
    *v28 = a5;
LABEL_40:
    v9 = *(_BYTE **)a1;
    v7 = *(_QWORD *)(a1 + 8);
  }
  while ( *(_QWORD *)a1 == v7 || *v9 != 81 );
  v33 = *(_BYTE *)(a1 + 778);
  *(_BYTE *)(a1 + 778) = 1;
  *(_QWORD *)a1 = v9 + 1;
  v34 = sub_E18BB0(a1);
  *(_BYTE *)(a1 + 778) = v33;
  v35 = v34;
  if ( !v34 )
    return 0;
  v36 = *(_BYTE **)a1;
  if ( *(_QWORD *)a1 == *(_QWORD *)(a1 + 8) || *v36 != 69 )
    return 0;
  *(_QWORD *)a1 = v36 + 1;
LABEL_47:
  v37 = sub_E11E80((_QWORD *)a1, (v62 - v13) >> 3, v7, v12, a5, a6);
  v39 = v38;
  result = sub_E0E790(a1 + 816, 40, v38, v40, v41, v42);
  if ( result )
  {
    *(_QWORD *)(result + 16) = v37;
    *(_WORD *)(result + 8) = 16427;
    v43 = *(_BYTE *)(result + 10);
    *(_QWORD *)(result + 24) = v39;
    *(_QWORD *)(result + 32) = v35;
    *(_BYTE *)(result + 10) = v43 & 0xF0 | 5;
    *(_QWORD *)result = &unk_49DFDE8;
  }
  return result;
}
