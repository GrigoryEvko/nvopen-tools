// Function: sub_1E798E0
// Address: 0x1e798e0
//
__int64 *__fastcall sub_1E798E0(
        __int64 *a1,
        __int64 *a2,
        __int64 a3,
        __int64 a4,
        __int64 a5,
        __int64 *a6,
        __int64 a7,
        __int64 a8)
{
  __int64 v8; // rax
  __int64 *v9; // r15
  __int64 *v10; // r13
  __int64 v12; // r11
  __int64 *v13; // r9
  __int64 *v14; // r11
  __int64 v15; // rbx
  __int64 v16; // r12
  __int64 *v17; // rax
  int v18; // r11d
  __int64 v19; // r8
  char *v20; // r9
  __int64 *v21; // r14
  __int64 v22; // r13
  char *v23; // r10
  size_t v24; // r10
  __int64 v25; // rax
  signed __int64 v26; // r14
  __int64 *v27; // rbx
  __int64 i; // r15
  __int64 v29; // rax
  __int64 v30; // rdi
  __int64 v31; // rdx
  unsigned __int64 v32; // rax
  __int64 *result; // rax
  __int64 v34; // rdi
  __int64 v35; // r14
  __int64 v36; // r13
  __int64 *v37; // rbx
  __int64 *v38; // r15
  __int64 *v39; // r14
  __int64 v40; // rax
  __int64 v41; // rdi
  __int64 v42; // rdx
  unsigned __int64 v43; // rax
  __int64 v44; // rdi
  __int64 v45; // r12
  __int64 *v46; // rsi
  __int64 *v47; // rdi
  size_t v48; // rdx
  __int64 *v49; // rax
  size_t v50; // rcx
  char *v51; // rax
  char *v52; // rax
  int v53; // [rsp+0h] [rbp-80h]
  char *v54; // [rsp+0h] [rbp-80h]
  __int64 v55; // [rsp+8h] [rbp-78h]
  int v56; // [rsp+8h] [rbp-78h]
  int v57; // [rsp+8h] [rbp-78h]
  int v58; // [rsp+8h] [rbp-78h]
  int v59; // [rsp+8h] [rbp-78h]
  __int64 *src; // [rsp+10h] [rbp-70h]
  void *srca; // [rsp+10h] [rbp-70h]
  void *srcb; // [rsp+10h] [rbp-70h]
  void *srcc; // [rsp+10h] [rbp-70h]
  __int64 srcd; // [rsp+10h] [rbp-70h]
  int srce; // [rsp+10h] [rbp-70h]
  void *srcf; // [rsp+10h] [rbp-70h]
  void *srcg; // [rsp+10h] [rbp-70h]
  int srch; // [rsp+10h] [rbp-70h]
  int srci; // [rsp+10h] [rbp-70h]
  __int64 *v70; // [rsp+18h] [rbp-68h]
  char *v71; // [rsp+18h] [rbp-68h]
  void *v72; // [rsp+18h] [rbp-68h]
  void *v73; // [rsp+18h] [rbp-68h]
  __int64 *v74; // [rsp+18h] [rbp-68h]
  __int64 *v75; // [rsp+18h] [rbp-68h]
  void *v76; // [rsp+18h] [rbp-68h]
  void *v77; // [rsp+18h] [rbp-68h]
  void *v78; // [rsp+18h] [rbp-68h]
  void *v79; // [rsp+18h] [rbp-68h]
  __int64 *dest; // [rsp+20h] [rbp-60h]
  __int64 *desta; // [rsp+20h] [rbp-60h]
  __int64 *destb; // [rsp+20h] [rbp-60h]
  __int64 *v83; // [rsp+28h] [rbp-58h]
  unsigned __int64 v84; // [rsp+28h] [rbp-58h]
  __int64 *v85; // [rsp+28h] [rbp-58h]
  __int64 v86; // [rsp+30h] [rbp-50h]
  __int64 v87; // [rsp+30h] [rbp-50h]
  __int64 v88; // [rsp+30h] [rbp-50h]
  unsigned __int64 v89; // [rsp+30h] [rbp-50h]
  __int64 v91; // [rsp+38h] [rbp-48h]
  _QWORD v92[7]; // [rsp+48h] [rbp-38h] BYREF

  v8 = a5;
  v9 = a2;
  v10 = a1;
  v12 = a8;
  if ( a7 <= a5 )
    v8 = a7;
  if ( a4 <= v8 )
    goto LABEL_22;
  if ( a7 >= a5 )
    goto LABEL_40;
  v13 = a2;
  v14 = a1;
  v15 = a5;
  dest = a6;
  v16 = a4;
  while ( 1 )
  {
    if ( v16 > v15 )
    {
      srce = (int)v14;
      v75 = v13;
      v22 = v16 / 2;
      v21 = &v14[v16 / 2];
      v49 = sub_1E78B90(v13, a3, v21, a8);
      v20 = (char *)v75;
      v18 = srce;
      v83 = v49;
      v19 = v49 - v75;
    }
    else
    {
      src = v13;
      v70 = v14;
      v83 = &v13[v15 / 2];
      v17 = sub_1E78AA0(v14, (__int64)v13, v83, a8);
      v18 = (int)v70;
      v19 = v15 / 2;
      v20 = (char *)src;
      v21 = v17;
      v22 = v17 - v70;
    }
    v16 -= v22;
    if ( v16 <= v19 || v19 > a7 )
    {
      if ( v16 > a7 )
      {
        srci = v18;
        v79 = (void *)v19;
        v52 = sub_1E78E60((char *)v21, v20, (char *)v83);
        v18 = srci;
        v19 = (__int64)v79;
        v23 = v52;
      }
      else
      {
        v23 = (char *)v83;
        if ( v16 )
        {
          v50 = v20 - (char *)v21;
          if ( v21 != (__int64 *)v20 )
          {
            v54 = v20;
            v58 = v18;
            srcf = (void *)v19;
            v76 = (void *)(v20 - (char *)v21);
            memmove(dest, v21, v20 - (char *)v21);
            v20 = v54;
            v18 = v58;
            v19 = (__int64)srcf;
            v50 = (size_t)v76;
          }
          if ( v83 != (__int64 *)v20 )
          {
            v59 = v18;
            srcg = (void *)v50;
            v77 = (void *)v19;
            memmove(v21, v20, (char *)v83 - v20);
            v18 = v59;
            v50 = (size_t)srcg;
            v19 = (__int64)v77;
          }
          v23 = (char *)v83 - v50;
          if ( v50 )
          {
            srch = v18;
            v78 = (void *)v19;
            v51 = (char *)memmove((char *)v83 - v50, dest, v50);
            v19 = (__int64)v78;
            v18 = srch;
            v23 = v51;
          }
        }
      }
    }
    else
    {
      v23 = (char *)v21;
      if ( v19 )
      {
        v24 = (char *)v83 - v20;
        if ( v83 != (__int64 *)v20 )
        {
          v53 = v18;
          v55 = v19;
          srca = (void *)((char *)v83 - v20);
          v71 = v20;
          memmove(dest, v20, (char *)v83 - v20);
          v18 = v53;
          v19 = v55;
          v24 = (size_t)srca;
          v20 = v71;
        }
        if ( v21 != (__int64 *)v20 )
        {
          v56 = v18;
          srcb = (void *)v24;
          v72 = (void *)v19;
          memmove((char *)v83 - (v20 - (char *)v21), v21, v20 - (char *)v21);
          v18 = v56;
          v24 = (size_t)srcb;
          v19 = (__int64)v72;
        }
        if ( v24 )
        {
          v57 = v18;
          srcc = (void *)v19;
          v73 = (void *)v24;
          memmove(v21, dest, v24);
          v18 = v57;
          v19 = (__int64)srcc;
          v24 = (size_t)v73;
        }
        v23 = (char *)v21 + v24;
      }
    }
    srcd = v19;
    v74 = (__int64 *)v23;
    sub_1E798E0(v18, (_DWORD)v21, (_DWORD)v23, v22, v19, (_DWORD)dest, a7, a8);
    v25 = a7;
    v15 -= srcd;
    if ( v15 <= a7 )
      v25 = v15;
    if ( v16 <= v25 )
    {
      v12 = a8;
      a6 = dest;
      v10 = v74;
      v9 = v83;
LABEL_22:
      v26 = (char *)v9 - (char *)v10;
      if ( v10 != v9 )
      {
        v86 = v12;
        memmove(a6, v10, (char *)v9 - (char *)v10);
        v12 = v86;
      }
      v92[0] = v12;
      desta = (__int64 *)((char *)a6 + v26);
      if ( a6 != (__int64 *)((char *)a6 + v26) && (__int64 *)a3 != v9 )
      {
        v27 = v9;
        for ( i = v12; ; i = v92[0] )
        {
          v34 = *(_QWORD *)(i + 280);
          v31 = *a6;
          v35 = *v27;
          if ( v34
            && (v87 = *a6, v29 = sub_1DDC3C0(v34, *v27), v30 = *(_QWORD *)(i + 280), v31 = v87, v84 = v29, v30)
            && (v32 = sub_1DDC3C0(v30, v87), v31 = v87, v84)
            && v32 )
          {
            if ( v84 < v32 )
              goto LABEL_31;
          }
          else if ( sub_1E78020((__int64)v92, v35, v31) )
          {
LABEL_31:
            result = (__int64 *)*v27;
            ++v10;
            ++v27;
            *(v10 - 1) = (__int64)result;
            if ( desta == a6 )
              return result;
            goto LABEL_32;
          }
          result = (__int64 *)*a6;
          ++v10;
          ++a6;
          *(v10 - 1) = (__int64)result;
          if ( desta == a6 )
            return result;
LABEL_32:
          if ( (__int64 *)a3 == v27 )
            break;
        }
      }
      result = desta;
      if ( desta == a6 )
        return result;
      v46 = a6;
      v47 = v10;
      v48 = (char *)desta - (char *)a6;
      return (__int64 *)memmove(v47, v46, v48);
    }
    if ( v15 <= a7 )
      break;
    v13 = v83;
    v14 = v74;
  }
  v12 = a8;
  a6 = dest;
  v10 = v74;
  v9 = v83;
LABEL_40:
  if ( (__int64 *)a3 != v9 )
  {
    v88 = v12;
    memmove(a6, v9, a3 - (_QWORD)v9);
    v12 = v88;
  }
  v92[0] = v12;
  result = (__int64 *)((char *)a6 + a3 - (_QWORD)v9);
  if ( v10 == v9 )
  {
    if ( a6 == result )
      return result;
    v48 = a3 - (_QWORD)v9;
    v47 = v9;
LABEL_70:
    v46 = a6;
    return (__int64 *)memmove(v47, v46, v48);
  }
  if ( a6 == result )
    return result;
  destb = v10;
  v85 = a6;
  v36 = v12;
  v37 = (__int64 *)(a3 - 8);
  v38 = v9 - 1;
  v39 = result - 1;
  while ( 2 )
  {
    v44 = *(_QWORD *)(v36 + 280);
    v42 = *v38;
    v45 = *v39;
    if ( v44 )
    {
      v91 = *v38;
      v40 = sub_1DDC3C0(v44, *v39);
      v41 = *(_QWORD *)(v36 + 280);
      v42 = v91;
      v89 = v40;
      if ( v41 )
      {
        v43 = sub_1DDC3C0(v41, v91);
        v42 = v91;
        if ( v89 )
        {
          if ( v43 )
          {
            if ( v89 < v43 )
              goto LABEL_49;
LABEL_54:
            result = (__int64 *)*v39;
            *v37 = *v39;
            if ( v85 == v39 )
              return result;
            --v39;
LABEL_51:
            v36 = v92[0];
            --v37;
            continue;
          }
        }
      }
    }
    break;
  }
  if ( !sub_1E78020((__int64)v92, v45, v42) )
    goto LABEL_54;
LABEL_49:
  result = (__int64 *)*v38;
  *v37 = *v38;
  if ( v38 != destb )
  {
    --v38;
    goto LABEL_51;
  }
  a6 = v85;
  if ( v85 != v39 + 1 )
  {
    v48 = (char *)(v39 + 1) - (char *)v85;
    v47 = (__int64 *)((char *)v37 - v48);
    goto LABEL_70;
  }
  return result;
}
