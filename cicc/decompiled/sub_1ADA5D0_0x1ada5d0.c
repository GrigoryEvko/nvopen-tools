// Function: sub_1ADA5D0
// Address: 0x1ada5d0
//
__int64 __fastcall sub_1ADA5D0(__int64 a1)
{
  __int64 v1; // r13
  __int64 v2; // rbx
  __int64 j; // r12
  __int64 v4; // rdi
  const char *v6; // rax
  size_t v7; // rdx
  size_t v8; // rbx
  char *v9; // r12
  __int64 *v10; // rax
  _QWORD *v11; // rax
  __int64 v12; // rax
  __int64 v13; // rbx
  __int64 i; // r14
  __int64 v15; // r8
  size_t v16; // rdx
  char *v17; // rsi
  __int64 v18; // r12
  size_t v19; // rdx
  char *v20; // rsi
  __int64 v21; // r15
  size_t v22; // rdx
  __int64 v23; // rax
  unsigned int v24; // esi
  __int64 v25; // rcx
  unsigned int v26; // edi
  __int64 v27; // r10
  __int64 *v28; // rdx
  __int64 v29; // r9
  __int64 v30; // rcx
  char *v31; // rsi
  size_t v32; // rdx
  size_t v33; // rax
  _QWORD *v34; // r8
  int v35; // ecx
  __int64 v36; // r8
  unsigned int v37; // r10d
  int v38; // edx
  __int64 *v39; // rdi
  __int64 v40; // r8
  int v41; // r8d
  int k; // r11d
  int v43; // esi
  __int64 *v44; // rcx
  __int64 *v45; // r11
  int v46; // edi
  __int64 *v47; // r8
  __int64 v48; // [rsp+10h] [rbp-80h]
  __int64 v50; // [rsp+20h] [rbp-70h]
  _QWORD *v51; // [rsp+20h] [rbp-70h]
  unsigned int v52; // [rsp+20h] [rbp-70h]
  unsigned int v53; // [rsp+28h] [rbp-68h]
  int v54; // [rsp+28h] [rbp-68h]
  __int64 v55; // [rsp+30h] [rbp-60h] BYREF
  __int64 *v56; // [rsp+38h] [rbp-58h] BYREF
  __int64 v57; // [rsp+40h] [rbp-50h] BYREF
  __int64 v58; // [rsp+48h] [rbp-48h]
  __int64 v59; // [rsp+50h] [rbp-40h]
  unsigned int v60; // [rsp+58h] [rbp-38h]

  v1 = a1 + 72;
  if ( !(unsigned __int8)sub_1CCAAC0(2, a1) )
  {
    v6 = sub_1649960(a1);
    v8 = v7;
    v9 = (char *)v6;
    v10 = (__int64 *)sub_15E0530(a1);
    v11 = (_QWORD *)sub_161FF10(v10, v9, v8);
    v12 = sub_1AD5AF0(a1, v11);
    v13 = *(_QWORD *)(a1 + 80);
    v48 = v12;
    if ( v1 == v13 )
      return sub_1CCAB50(2, a1);
    while ( 1 )
    {
      if ( !v13 )
LABEL_113:
        BUG();
      i = *(_QWORD *)(v13 + 24);
      if ( i != v13 + 16 )
        break;
      v13 = *(_QWORD *)(v13 + 8);
      if ( v1 == v13 )
        return sub_1CCAB50(2, a1);
    }
    if ( v1 == v13 )
      return sub_1CCAB50(2, a1);
    while ( 1 )
    {
      v15 = i - 24;
      if ( !i )
        v15 = 0;
      v16 = 0;
      v17 = off_4CD4970[0];
      v18 = v15;
      if ( off_4CD4970[0] )
      {
        v17 = off_4CD4970[0];
        v16 = strlen(off_4CD4970[0]);
      }
      if ( *(_QWORD *)(v18 + 48) || *(__int16 *)(v18 + 18) < 0 )
      {
        if ( sub_1625940(v18, v17, v16) )
          goto LABEL_20;
        v17 = off_4CD4970[0];
      }
      v19 = 0;
      if ( v17 )
        v19 = strlen(v17);
      sub_1626100(v18, v17, v19, v48);
LABEL_20:
      for ( i = *(_QWORD *)(i + 8); i == v13 - 24 + 40; i = *(_QWORD *)(v13 + 24) )
      {
        v13 = *(_QWORD *)(v13 + 8);
        if ( v1 == v13 )
          return sub_1CCAB50(2, a1);
        if ( !v13 )
          goto LABEL_113;
      }
      if ( v1 == v13 )
        return sub_1CCAB50(2, a1);
    }
  }
  v2 = *(_QWORD *)(a1 + 80);
  v57 = 0;
  v58 = 0;
  v59 = 0;
  v60 = 0;
  if ( v1 == v2 )
  {
LABEL_6:
    v4 = 0;
  }
  else
  {
    while ( 1 )
    {
      if ( !v2 )
        goto LABEL_113;
      j = *(_QWORD *)(v2 + 24);
      if ( j != v2 + 16 )
        break;
      v2 = *(_QWORD *)(v2 + 8);
      if ( v1 == v2 )
        goto LABEL_6;
    }
    if ( v1 != v2 )
    {
      while ( 1 )
      {
        v20 = off_4CD4970[0];
        v21 = j - 24;
        if ( !j )
          v21 = 0;
        v22 = 0;
        if ( off_4CD4970[0] )
        {
          v20 = off_4CD4970[0];
          v22 = strlen(off_4CD4970[0]);
        }
        if ( *(_QWORD *)(v21 + 48) || *(__int16 *)(v21 + 18) < 0 )
        {
          v23 = sub_1625940(v21, v20, v22);
          v55 = v23;
          if ( v23 )
            break;
        }
LABEL_46:
        for ( j = *(_QWORD *)(j + 8); j == v2 - 24 + 40; j = *(_QWORD *)(v2 + 24) )
        {
          v2 = *(_QWORD *)(v2 + 8);
          if ( v1 == v2 )
            goto LABEL_52;
          if ( !v2 )
            goto LABEL_113;
        }
        if ( v1 == v2 )
          goto LABEL_52;
      }
      v24 = v60;
      v25 = v58;
      if ( v60 )
      {
        v26 = v60 - 1;
        v53 = v60 - 1;
        v27 = (v60 - 1) & (((unsigned int)v23 >> 9) ^ ((unsigned int)v23 >> 4));
        v28 = (__int64 *)(v58 + 16 * v27);
        v29 = *v28;
        if ( *v28 == v23 )
        {
          if ( v28 != (__int64 *)(v58 + 16LL * v60) )
          {
LABEL_42:
            v30 = v28[1];
LABEL_43:
            v31 = off_4CD4970[0];
            v32 = 0;
            if ( off_4CD4970[0] )
            {
              v50 = v30;
              v33 = strlen(off_4CD4970[0]);
              v30 = v50;
              v32 = v33;
            }
            sub_1626100(v21, v31, v32, v30);
            goto LABEL_46;
          }
        }
        else
        {
          v52 = (v60 - 1) & (((unsigned int)v23 >> 9) ^ ((unsigned int)v23 >> 4));
          v36 = *v28;
          for ( k = 1; ; ++k )
          {
            if ( v36 == -8 )
            {
              v34 = *(_QWORD **)(v23 - 8LL * *(unsigned int *)(v23 + 8));
              LODWORD(v27) = v26 & (((unsigned int)v23 >> 4) ^ ((unsigned int)v23 >> 9));
              v28 = (__int64 *)(v58 + 16LL * (unsigned int)v27);
              v29 = *v28;
              goto LABEL_64;
            }
            v52 = v26 & (k + v52);
            v36 = *(_QWORD *)(v58 + 16LL * v52);
            if ( v23 == v36 )
              break;
          }
          if ( v58 + 16LL * v52 != v58 + 16LL * v60 )
            goto LABEL_73;
        }
        v34 = *(_QWORD **)(v23 - 8LL * *(unsigned int *)(v23 + 8));
LABEL_64:
        if ( v29 == v23 )
        {
LABEL_65:
          v28[1] = sub_1AD5AF0(a1, v34);
          v24 = v60;
          if ( v60 )
          {
            v23 = v55;
            v25 = v58;
            v53 = v60 - 1;
            v27 = (v60 - 1) & (((unsigned int)v55 >> 9) ^ ((unsigned int)v55 >> 4));
            v28 = (__int64 *)(v58 + 16 * v27);
            v29 = *v28;
            if ( v55 == *v28 )
              goto LABEL_42;
LABEL_73:
            v41 = 1;
            v39 = 0;
            while ( v29 != -8 )
            {
              if ( v29 != -16 || v39 )
                v28 = v39;
              LODWORD(v27) = v53 & (v41 + v27);
              v46 = v41 + 1;
              v47 = (__int64 *)(v25 + 16LL * (unsigned int)v27);
              v29 = *v47;
              if ( *v47 == v23 )
              {
                v30 = v47[1];
                goto LABEL_43;
              }
              v41 = v46;
              v39 = v28;
              v28 = (__int64 *)(v25 + 16LL * (unsigned int)v27);
            }
            if ( !v39 )
              v39 = v28;
            ++v57;
            v38 = v59 + 1;
            if ( 4 * ((int)v59 + 1) < 3 * v24 )
            {
              if ( v24 - (v38 + HIDWORD(v59)) <= v24 >> 3 )
              {
                sub_15AC120((__int64)&v57, v24);
                sub_1AD5D50((__int64)&v57, &v55, &v56);
                v39 = v56;
                v23 = v55;
                v38 = v59 + 1;
              }
LABEL_69:
              LODWORD(v59) = v38;
              if ( *v39 != -8 )
                --HIDWORD(v59);
              *v39 = v23;
              v30 = 0;
              v39[1] = 0;
              goto LABEL_43;
            }
          }
          else
          {
            ++v57;
          }
          sub_15AC120((__int64)&v57, 2 * v24);
          if ( !v60 )
          {
            LODWORD(v59) = v59 + 1;
            BUG();
          }
          v23 = v55;
          v37 = (v60 - 1) & (((unsigned int)v55 >> 9) ^ ((unsigned int)v55 >> 4));
          v38 = v59 + 1;
          v39 = (__int64 *)(v58 + 16LL * v37);
          v40 = *v39;
          if ( *v39 != v55 )
          {
            v43 = 1;
            v44 = 0;
            while ( v40 != -8 )
            {
              if ( !v44 && v40 == -16 )
                v44 = v39;
              v37 = (v60 - 1) & (v43 + v37);
              v39 = (__int64 *)(v58 + 16LL * v37);
              v40 = *v39;
              if ( v55 == *v39 )
                goto LABEL_69;
              ++v43;
            }
            if ( v44 )
              v39 = v44;
          }
          goto LABEL_69;
        }
        v54 = 1;
        v45 = 0;
        while ( v29 != -8 )
        {
          if ( v29 == -16 )
          {
            if ( v45 )
              v28 = v45;
            v45 = v28;
          }
          LODWORD(v27) = v26 & (v54 + v27);
          v28 = (__int64 *)(v58 + 16LL * (unsigned int)v27);
          v29 = *v28;
          if ( v23 == *v28 )
            goto LABEL_65;
          ++v54;
        }
        if ( v45 )
          v28 = v45;
        ++v57;
        v35 = v59 + 1;
        if ( 4 * ((int)v59 + 1) < 3 * v60 )
        {
          if ( v60 - HIDWORD(v59) - v35 > v60 >> 3 )
            goto LABEL_58;
          v51 = v34;
LABEL_57:
          sub_15AC120((__int64)&v57, v24);
          sub_1AD5D50((__int64)&v57, &v55, &v56);
          v28 = v56;
          v23 = v55;
          v34 = v51;
          v35 = v59 + 1;
LABEL_58:
          LODWORD(v59) = v35;
          if ( *v28 != -8 )
            --HIDWORD(v59);
          *v28 = v23;
          v28[1] = 0;
          goto LABEL_65;
        }
      }
      else
      {
        v34 = *(_QWORD **)(v23 - 8LL * *(unsigned int *)(v23 + 8));
        ++v57;
      }
      v51 = v34;
      v24 = 2 * v60;
      goto LABEL_57;
    }
LABEL_52:
    v4 = v58;
  }
  return j___libc_free_0(v4);
}
