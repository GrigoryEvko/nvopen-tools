// Function: sub_29F64D0
// Address: 0x29f64d0
//
__int64 __fastcall sub_29F64D0(__int64 a1, __int64 a2, __int64 a3, __int64 a4, char *a5, __int64 a6)
{
  __int64 result; // rax
  __int64 v8; // r10
  char *v9; // rdx
  __int64 v10; // r15
  char *v11; // r9
  __int64 v12; // r8
  char *v13; // rcx
  char *v14; // r11
  char *v15; // rdi
  char *v16; // rsi
  char *v17; // rax
  unsigned __int8 *v18; // rax
  __int64 v19; // rsi
  bool v20; // cc
  unsigned __int8 *v21; // rbx
  __int64 v22; // rdx
  __int64 v23; // rcx
  __int64 v24; // r8
  __int64 v25; // r9
  char *v26; // rdi
  char *v27; // rsi
  __int64 v28; // r9
  unsigned __int64 v29; // r15
  char **v30; // rbx
  char *v31; // r10
  char *v32; // rax
  __int64 v33; // rcx
  char *v34; // rdi
  char *v35; // rdx
  __int64 v36; // rdi
  char *v37; // r8
  char *v38; // rax
  char *v39; // r11
  __int64 v40; // rdx
  __int64 v41; // rcx
  __int64 v42; // r8
  __int64 v43; // r9
  __int64 v44; // rbx
  __int64 v45; // rcx
  __int64 v46; // r8
  __int64 v47; // r9
  __int64 v48; // rsi
  char *v49; // rax
  char *v50; // rbx
  char *v51; // r10
  __int64 v52; // rdx
  __int64 v53; // rcx
  __int64 v54; // r8
  __int64 v55; // r9
  char **v56; // [rsp+0h] [rbp-F0h]
  __int64 v57; // [rsp+8h] [rbp-E8h]
  __int64 v58; // [rsp+10h] [rbp-E0h]
  char *v59; // [rsp+20h] [rbp-D0h] BYREF
  __int64 v60; // [rsp+28h] [rbp-C8h]
  __int64 v61; // [rsp+30h] [rbp-C0h]
  _BYTE v62[184]; // [rsp+38h] [rbp-B8h] BYREF

  result = a2 - a1;
  v58 = a2;
  v57 = a3;
  if ( a2 - a1 <= 2432 )
    return result;
  if ( !a3 )
  {
LABEL_50:
    v44 = v58;
    sub_29F4AC0(a1, v58, a3, a4, a5, a6);
    do
    {
      v44 -= 152;
      sub_29F49A0((char **)a1, v44, v44, v45, v46, v47);
      result = v44 - a1;
    }
    while ( v44 - a1 > 152 );
    return result;
  }
  v56 = (char **)(a1 + 152);
  while ( 2 )
  {
    v8 = *(_QWORD *)(a1 + 160);
    --v57;
    v9 = *(char **)(a1 + 152);
    v10 = a1 + 152 * ((__int64)(0x86BCA1AF286BCA1BLL * (result >> 3)) >> 1);
    v11 = *(char **)(v10 + 8);
    v12 = (__int64)&v9[v8];
    v13 = *(char **)v10;
    v14 = &v11[(_QWORD)v9];
    v15 = &v11[*(_QWORD *)v10];
    v16 = *(char **)v10;
    if ( (__int64)v11 >= v8 )
      v14 = &v9[v8];
    if ( v9 != v14 )
    {
      v17 = *(char **)(a1 + 152);
      while ( *v17 >= *v16 )
      {
        if ( *v17 > *v16 )
          goto LABEL_54;
        ++v17;
        ++v16;
        if ( v14 == v17 )
          goto LABEL_53;
      }
LABEL_11:
      v18 = *(unsigned __int8 **)(v58 - 152);
      v19 = *(_QWORD *)(v58 - 144);
      v20 = (__int64)v11 <= v19;
      v21 = &v18[v19];
      v11 = (char *)v18;
      if ( !v20 )
        v15 = &v13[v19];
      if ( v13 == v15 )
      {
LABEL_65:
        if ( v21 == (unsigned __int8 *)v11 )
        {
LABEL_66:
          v13 = &v9[v19];
          if ( v19 < v8 )
            v12 = (__int64)&v9[v19];
          for ( ; (char *)v12 != v9; ++v18 )
          {
            v13 = (char *)*v18;
            if ( *v9 < (char)v13 )
              goto LABEL_73;
            if ( *v9 > (char)v13 )
              goto LABEL_61;
            ++v9;
          }
          if ( v21 != v18 )
          {
LABEL_73:
            v59 = v62;
            v60 = 0;
            v61 = 128;
            if ( *(_QWORD *)(a1 + 8) )
              sub_29F3DD0((__int64)&v59, (char **)a1, (__int64)v9, (__int64)v13, v12, (__int64)v11);
            v10 = v58 - 152;
            goto LABEL_20;
          }
LABEL_61:
          v59 = v62;
          v60 = 0;
          v61 = 128;
          if ( *(_QWORD *)(a1 + 8) )
            sub_29F3DD0((__int64)&v59, (char **)a1, (__int64)v9, (__int64)v13, v12, (__int64)v11);
          sub_29F3DD0(a1, v56, (__int64)v9, (__int64)v13, v12, (__int64)v11);
          sub_29F3DD0((__int64)v56, &v59, v52, v53, v54, v55);
          v26 = v59;
          if ( v59 == v62 )
            goto LABEL_22;
          goto LABEL_21;
        }
      }
      else
      {
        while ( *v13 >= *v11 )
        {
          if ( *v13 > *v11 )
            goto LABEL_66;
          ++v13;
          ++v11;
          if ( v15 == v13 )
            goto LABEL_65;
        }
      }
      goto LABEL_18;
    }
LABEL_53:
    if ( v16 != v15 )
      goto LABEL_11;
LABEL_54:
    v48 = *(_QWORD *)(v58 - 144);
    v49 = *(char **)(v58 - 152);
    v20 = v48 < v8;
    v50 = &v49[v48];
    v51 = v49;
    if ( v20 )
      v12 = (__int64)&v9[v48];
    if ( v9 != (char *)v12 )
    {
      while ( *v9 >= *v51 )
      {
        if ( *v9 > *v51 )
          goto LABEL_77;
        ++v9;
        ++v51;
        if ( (char *)v12 == v9 )
          goto LABEL_76;
      }
      goto LABEL_61;
    }
LABEL_76:
    if ( v51 != v50 )
      goto LABEL_61;
LABEL_77:
    v9 = &v13[v48];
    if ( (__int64)v11 > v48 )
      v15 = &v13[v48];
    if ( v13 != v15 )
    {
      while ( *v13 >= *v49 )
      {
        if ( *v13 > *v49 )
          goto LABEL_18;
        ++v13;
        ++v49;
        if ( v15 == v13 )
          goto LABEL_85;
      }
      goto LABEL_73;
    }
LABEL_85:
    if ( v49 != v50 )
      goto LABEL_73;
LABEL_18:
    v59 = v62;
    v60 = 0;
    v61 = 128;
    if ( *(_QWORD *)(a1 + 8) )
      sub_29F3DD0((__int64)&v59, (char **)a1, (__int64)v9, (__int64)v13, v12, (__int64)v11);
LABEL_20:
    sub_29F3DD0(a1, (char **)v10, (__int64)v9, (__int64)v13, v12, (__int64)v11);
    sub_29F3DD0(v10, &v59, v22, v23, v24, v25);
    v26 = v59;
    if ( v59 != v62 )
LABEL_21:
      _libc_free((unsigned __int64)v26);
LABEL_22:
    v27 = *(char **)a1;
    v28 = *(_QWORD *)(a1 + 8);
    v29 = (unsigned __int64)v56;
    v30 = (char **)v58;
    v31 = (char *)(*(_QWORD *)a1 + v28);
    while ( 1 )
    {
      v32 = *(char **)v29;
      v33 = *(_QWORD *)(v29 + 8);
      v34 = (char *)(*(_QWORD *)v29 + v28);
      if ( v33 <= v28 )
        v34 = (char *)(*(_QWORD *)v29 + v33);
      v35 = v27;
      if ( v32 == v34 )
        break;
      while ( *v32 >= *v35 )
      {
        if ( *v32 > *v35 )
          goto LABEL_32;
        ++v32;
        ++v35;
        if ( v34 == v32 )
          goto LABEL_46;
      }
LABEL_45:
      v29 += 152LL;
    }
LABEL_46:
    if ( v35 != v31 )
      goto LABEL_45;
    do
    {
      while ( 1 )
      {
LABEL_32:
        v36 = (__int64)*(v30 - 18);
        v30 -= 19;
        v37 = *v30;
        v38 = &v27[v36];
        v39 = *v30;
        if ( v36 >= v28 )
          v38 = v31;
        if ( v38 == v27 )
          break;
        v35 = v27;
        while ( *v35 >= *v39 )
        {
          if ( *v35 > *v39 )
            goto LABEL_39;
          ++v35;
          ++v39;
          if ( v38 == v35 )
            goto LABEL_31;
        }
      }
LABEL_31:
      ;
    }
    while ( v39 != &v37[v36] );
LABEL_39:
    if ( v29 < (unsigned __int64)v30 )
    {
      v59 = v62;
      v60 = 0;
      v61 = 128;
      if ( v33 )
        sub_29F3DD0((__int64)&v59, (char **)v29, (__int64)v35, v33, (__int64)v37, v28);
      sub_29F3DD0(v29, v30, (__int64)v35, v33, (__int64)v37, v28);
      sub_29F3DD0((__int64)v30, &v59, v40, v41, v42, v43);
      if ( v59 != v62 )
        _libc_free((unsigned __int64)v59);
      v28 = *(_QWORD *)(a1 + 8);
      v27 = *(char **)a1;
      v31 = (char *)(*(_QWORD *)a1 + v28);
      goto LABEL_45;
    }
    sub_29F64D0(v29, v58, v57);
    result = v29 - a1;
    if ( (__int64)(v29 - a1) > 2432 )
    {
      v58 = v29;
      if ( !v57 )
        goto LABEL_50;
      continue;
    }
    return result;
  }
}
