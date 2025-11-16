// Function: sub_29F8150
// Address: 0x29f8150
//
void __fastcall sub_29F8150(__int64 a1, unsigned __int64 a2, __int64 a3, __int64 a4, char *a5, __int64 a6)
{
  __int64 v6; // rax
  __int64 v8; // r10
  char *v9; // rdx
  __int64 v10; // r12
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
  unsigned __int8 *v21; // r13
  __int64 v22; // rdx
  __int64 v23; // rcx
  __int64 v24; // r8
  __int64 v25; // r9
  char *v26; // rdi
  char *v27; // rsi
  __int64 v28; // r9
  unsigned __int64 v29; // r13
  char **v30; // r12
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
  __int64 v44; // rdx
  __int64 v45; // rcx
  __int64 v46; // r8
  __int64 v47; // r9
  __int64 v48; // r12
  __int64 v49; // rdx
  __int64 v50; // rcx
  char *v51; // r8
  __int64 v52; // r9
  bool v53; // zf
  char *v54; // rax
  __int64 v55; // rsi
  char *v56; // r13
  char *v57; // r10
  __int64 v58; // rdx
  __int64 v59; // rcx
  __int64 v60; // r8
  __int64 v61; // r9
  char **v62; // [rsp+0h] [rbp-110h]
  __int64 v63; // [rsp+8h] [rbp-108h]
  unsigned __int64 v64; // [rsp+10h] [rbp-100h]
  char *v65; // [rsp+20h] [rbp-F0h] BYREF
  __int64 v66; // [rsp+28h] [rbp-E8h]
  __int64 v67; // [rsp+30h] [rbp-E0h]
  _BYTE v68[72]; // [rsp+38h] [rbp-D8h] BYREF
  char *v69; // [rsp+80h] [rbp-90h] BYREF
  __int64 v70; // [rsp+88h] [rbp-88h]
  __int64 v71; // [rsp+90h] [rbp-80h]
  _BYTE v72[120]; // [rsp+98h] [rbp-78h] BYREF

  v6 = a2 - a1;
  v64 = a2;
  v63 = a3;
  if ( (__int64)(a2 - a1) <= 1408 )
    return;
  if ( !a3 )
  {
LABEL_50:
    sub_29F7E10(a1, v64, v64, a4, a5, a6);
    v48 = v64;
    do
    {
      v48 -= 88;
      v66 = 0;
      v53 = *(_QWORD *)(v48 + 8) == 0;
      v67 = 64;
      v65 = v68;
      if ( !v53 )
        sub_29F3DD0((__int64)&v65, (char **)v48, v44, v45, v46, v47);
      sub_29F3DD0(v48, (char **)a1, v44, v45, v46, v47);
      v69 = v72;
      v70 = 0;
      v71 = 64;
      if ( v66 )
        sub_29F3DD0((__int64)&v69, &v65, v49, v50, (__int64)v51, v52);
      sub_29F4C70(a1, 0, (char *)(0x2E8BA2E8BA2E8BA3LL * ((v48 - a1) >> 3)), (__int64)&v69, v51, v52);
      if ( v69 != v72 )
        _libc_free((unsigned __int64)v69);
      if ( v65 != v68 )
        _libc_free((unsigned __int64)v65);
    }
    while ( v48 - a1 > 88 );
    return;
  }
  v62 = (char **)(a1 + 88);
  while ( 2 )
  {
    v8 = *(_QWORD *)(a1 + 96);
    --v63;
    v9 = *(char **)(a1 + 88);
    v10 = a1 + 88 * ((0x2E8BA2E8BA2E8BA3LL * (v6 >> 3)) >> 1);
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
      v17 = *(char **)(a1 + 88);
      while ( *v17 >= *v16 )
      {
        if ( *v17 > *v16 )
          goto LABEL_61;
        ++v17;
        ++v16;
        if ( v14 == v17 )
          goto LABEL_60;
      }
LABEL_11:
      v18 = *(unsigned __int8 **)(v64 - 88);
      v19 = *(_QWORD *)(v64 - 80);
      v20 = v19 < (__int64)v11;
      v21 = &v18[v19];
      v11 = (char *)v18;
      if ( v20 )
        v15 = &v13[v19];
      if ( v13 == v15 )
      {
LABEL_83:
        if ( v21 == (unsigned __int8 *)v11 )
        {
LABEL_84:
          v13 = &v9[v19];
          if ( v19 < v8 )
            v12 = (__int64)&v9[v19];
          for ( ; (char *)v12 != v9; ++v18 )
          {
            v13 = (char *)*v18;
            if ( *v9 < (char)v13 )
              goto LABEL_80;
            if ( *v9 > (char)v13 )
              goto LABEL_68;
            ++v9;
          }
          if ( v21 != v18 )
          {
LABEL_80:
            v69 = v72;
            v70 = 0;
            v71 = 64;
            if ( *(_QWORD *)(a1 + 8) )
              sub_29F3DD0((__int64)&v69, (char **)a1, (__int64)v9, (__int64)v13, v12, (__int64)v11);
            v10 = v64 - 88;
            goto LABEL_20;
          }
LABEL_68:
          v69 = v72;
          v70 = 0;
          v71 = 64;
          if ( *(_QWORD *)(a1 + 8) )
            sub_29F3DD0((__int64)&v69, (char **)a1, (__int64)v9, (__int64)v13, v12, (__int64)v11);
          sub_29F3DD0(a1, v62, (__int64)v9, (__int64)v13, v12, (__int64)v11);
          sub_29F3DD0((__int64)v62, &v69, v58, v59, v60, v61);
          v26 = v69;
          if ( v69 == v72 )
            goto LABEL_22;
          goto LABEL_21;
        }
      }
      else
      {
        while ( *v13 >= *v11 )
        {
          if ( *v13 > *v11 )
            goto LABEL_84;
          ++v13;
          ++v11;
          if ( v15 == v13 )
            goto LABEL_83;
        }
      }
      goto LABEL_18;
    }
LABEL_60:
    if ( v15 != v16 )
      goto LABEL_11;
LABEL_61:
    v54 = *(char **)(v64 - 88);
    v55 = *(_QWORD *)(v64 - 80);
    v20 = v55 < v8;
    v56 = &v54[v55];
    v57 = v54;
    if ( v20 )
      v12 = (__int64)&v9[v55];
    if ( v9 != (char *)v12 )
    {
      while ( *v9 >= *v57 )
      {
        if ( *v9 > *v57 )
          goto LABEL_73;
        ++v9;
        ++v57;
        if ( (char *)v12 == v9 )
          goto LABEL_72;
      }
      goto LABEL_68;
    }
LABEL_72:
    if ( v57 != v56 )
      goto LABEL_68;
LABEL_73:
    v9 = &v13[v55];
    if ( v55 < (__int64)v11 )
      v15 = &v13[v55];
    if ( v13 != v15 )
    {
      while ( *v13 >= *v54 )
      {
        if ( *v13 > *v54 )
          goto LABEL_18;
        ++v13;
        ++v54;
        if ( v15 == v13 )
          goto LABEL_95;
      }
      goto LABEL_80;
    }
LABEL_95:
    if ( v56 != v54 )
      goto LABEL_80;
LABEL_18:
    v69 = v72;
    v70 = 0;
    v71 = 64;
    if ( *(_QWORD *)(a1 + 8) )
      sub_29F3DD0((__int64)&v69, (char **)a1, (__int64)v9, (__int64)v13, v12, (__int64)v11);
LABEL_20:
    sub_29F3DD0(a1, (char **)v10, (__int64)v9, (__int64)v13, v12, (__int64)v11);
    sub_29F3DD0(v10, &v69, v22, v23, v24, v25);
    v26 = v69;
    if ( v69 != v72 )
LABEL_21:
      _libc_free((unsigned __int64)v26);
LABEL_22:
    v27 = *(char **)a1;
    v28 = *(_QWORD *)(a1 + 8);
    v29 = (unsigned __int64)v62;
    v30 = (char **)v64;
    v31 = (char *)(*(_QWORD *)a1 + v28);
    while ( 1 )
    {
      v32 = *(char **)v29;
      v33 = *(_QWORD *)(v29 + 8);
      v34 = (char *)(*(_QWORD *)v29 + v28);
      if ( v28 >= v33 )
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
      v29 += 88LL;
    }
LABEL_46:
    if ( v35 != v31 )
      goto LABEL_45;
    do
    {
      while ( 1 )
      {
LABEL_32:
        v36 = (__int64)*(v30 - 10);
        v30 -= 11;
        v37 = *v30;
        v38 = &v27[v36];
        v39 = *v30;
        if ( v28 <= v36 )
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
      v69 = v72;
      v70 = 0;
      v71 = 64;
      if ( v33 )
        sub_29F3DD0((__int64)&v69, (char **)v29, (__int64)v35, v33, (__int64)v37, v28);
      sub_29F3DD0(v29, v30, (__int64)v35, v33, (__int64)v37, v28);
      sub_29F3DD0((__int64)v30, &v69, v40, v41, v42, v43);
      if ( v69 != v72 )
        _libc_free((unsigned __int64)v69);
      v28 = *(_QWORD *)(a1 + 8);
      v27 = *(char **)a1;
      v31 = (char *)(*(_QWORD *)a1 + v28);
      goto LABEL_45;
    }
    sub_29F8150(v29, v64, v63);
    v6 = v29 - a1;
    if ( (__int64)(v29 - a1) > 1408 )
    {
      v64 = v29;
      if ( !v63 )
        goto LABEL_50;
      continue;
    }
    break;
  }
}
