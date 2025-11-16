// Function: sub_295AE70
// Address: 0x295ae70
//
__int64 __fastcall sub_295AE70(
        char *a1,
        char *a2,
        __int64 a3,
        __int64 a4,
        __int64 a5,
        char *a6,
        __int64 a7,
        __int64 a8)
{
  __int64 result; // rax
  char *v9; // r13
  char *v10; // r12
  char *v11; // rbx
  __int64 v12; // r15
  char *v13; // r13
  __int64 v14; // rbx
  __int64 *v15; // r9
  char *v16; // rax
  __int64 v17; // r8
  char *v18; // r9
  char *v19; // r10
  __int64 v20; // r11
  size_t v21; // rcx
  char *v22; // rax
  signed __int64 v23; // r15
  char *v24; // r11
  int v25; // ecx
  __int64 v26; // r8
  __int64 v27; // rsi
  __int64 v28; // r10
  int v29; // ecx
  unsigned int v30; // edx
  __int64 *v31; // rax
  __int64 v32; // r15
  _QWORD *v33; // rdx
  unsigned int v34; // edi
  __int64 *v35; // rdx
  __int64 v36; // r15
  _QWORD **v37; // rdx
  _QWORD *v38; // rdx
  unsigned int i; // ecx
  int v40; // edx
  char *v41; // r10
  char *v42; // r12
  char *v43; // r10
  char *v44; // rbx
  int v45; // ecx
  __int64 v46; // rdi
  __int64 v47; // r8
  __int64 v48; // r11
  int v49; // ecx
  unsigned int v50; // edx
  __int64 *v51; // rax
  __int64 v52; // r15
  _QWORD *v53; // rdx
  unsigned int v54; // esi
  __int64 *v55; // rdx
  __int64 v56; // r15
  _QWORD **v57; // rdx
  _QWORD *v58; // rdx
  unsigned int j; // ecx
  int v60; // edx
  int v61; // eax
  char *v62; // rsi
  char *v63; // rdi
  size_t v64; // rdx
  int v65; // eax
  __int64 *v66; // rax
  size_t v67; // rcx
  unsigned int v68; // eax
  char *v69; // r10
  char *v70; // rax
  int v71; // edi
  int v72; // esi
  char *v73; // [rsp+0h] [rbp-70h]
  char *v74; // [rsp+0h] [rbp-70h]
  __int64 v75; // [rsp+8h] [rbp-68h]
  size_t v76; // [rsp+8h] [rbp-68h]
  __int64 v77; // [rsp+8h] [rbp-68h]
  int v78; // [rsp+8h] [rbp-68h]
  __int64 v79; // [rsp+8h] [rbp-68h]
  int v80; // [rsp+10h] [rbp-60h]
  __int64 v81; // [rsp+10h] [rbp-60h]
  __int64 v82; // [rsp+10h] [rbp-60h]
  int v83; // [rsp+10h] [rbp-60h]
  size_t v84; // [rsp+10h] [rbp-60h]
  __int64 v85; // [rsp+10h] [rbp-60h]
  int v86; // [rsp+10h] [rbp-60h]
  size_t v87; // [rsp+18h] [rbp-58h]
  int v88; // [rsp+18h] [rbp-58h]
  int v89; // [rsp+18h] [rbp-58h]
  __int64 v90; // [rsp+18h] [rbp-58h]
  size_t v91; // [rsp+18h] [rbp-58h]
  __int64 v92; // [rsp+18h] [rbp-58h]
  int v93; // [rsp+18h] [rbp-58h]
  int v94; // [rsp+18h] [rbp-58h]
  __int64 *v96; // [rsp+28h] [rbp-48h]
  char *src; // [rsp+30h] [rbp-40h]
  char *srca; // [rsp+30h] [rbp-40h]
  char *srcb; // [rsp+30h] [rbp-40h]
  char *srcc; // [rsp+30h] [rbp-40h]
  void *srcd; // [rsp+30h] [rbp-40h]
  char *srce; // [rsp+30h] [rbp-40h]
  int srcf; // [rsp+30h] [rbp-40h]
  char *dest; // [rsp+38h] [rbp-38h]
  int desta; // [rsp+38h] [rbp-38h]
  int destb; // [rsp+38h] [rbp-38h]

  result = a5;
  v9 = a1;
  v10 = a2;
  v11 = (char *)a3;
  if ( a7 <= a5 )
    result = a7;
  if ( a4 <= result )
    goto LABEL_22;
  v12 = a5;
  if ( a7 >= a5 )
    goto LABEL_44;
  dest = a1;
  v13 = a6;
  v14 = a4;
  v15 = (__int64 *)a2;
  while ( 1 )
  {
    src = (char *)v15;
    if ( v12 < v14 )
    {
      v66 = sub_2958170(v15, a3, (__int64 *)&dest[8 * (v14 / 2)], a8);
      v18 = src;
      v19 = &dest[8 * (v14 / 2)];
      v96 = v66;
      v20 = v14 / 2;
      v17 = ((char *)v66 - src) >> 3;
    }
    else
    {
      v96 = &v15[v12 / 2];
      v16 = (char *)sub_2958320(dest, (__int64)v15, v96, a8);
      v17 = v12 / 2;
      v18 = src;
      v19 = v16;
      v20 = (v16 - dest) >> 3;
    }
    v14 -= v20;
    if ( v14 <= v17 || v17 > a7 )
    {
      if ( v14 > a7 )
      {
        v79 = v17;
        v86 = v20;
        v94 = (int)v19;
        v70 = sub_295A4F0(v19, v18, (char *)v96);
        v17 = v79;
        LODWORD(v20) = v86;
        srca = v70;
        LODWORD(v19) = v94;
      }
      else
      {
        srca = (char *)v96;
        if ( v14 )
        {
          v67 = v18 - v19;
          if ( v18 != v19 )
          {
            v74 = v18;
            v77 = v17;
            v83 = v20;
            v91 = v18 - v19;
            srce = v19;
            memmove(v13, v19, v18 - v19);
            v18 = v74;
            v17 = v77;
            LODWORD(v20) = v83;
            v67 = v91;
            v19 = srce;
          }
          if ( v18 != (char *)v96 )
          {
            v84 = v67;
            v92 = v17;
            srcf = v20;
            v68 = (unsigned int)memmove(v19, v18, (char *)v96 - v18);
            v67 = v84;
            v17 = v92;
            LODWORD(v20) = srcf;
            LODWORD(v19) = v68;
          }
          srca = (char *)v96 - v67;
          if ( v67 )
          {
            v78 = (int)v19;
            v85 = v17;
            v93 = v20;
            memmove((char *)v96 - v67, v13, v67);
            LODWORD(v20) = v93;
            v17 = v85;
            LODWORD(v19) = v78;
          }
        }
      }
    }
    else
    {
      srca = v19;
      if ( v17 )
      {
        v21 = (char *)v96 - v18;
        if ( v18 != (char *)v96 )
        {
          v73 = v19;
          v75 = v17;
          v80 = v20;
          v87 = (char *)v96 - v18;
          srcb = v18;
          memmove(v13, v18, (char *)v96 - v18);
          v19 = v73;
          v17 = v75;
          LODWORD(v20) = v80;
          v21 = v87;
          v18 = srcb;
        }
        if ( v18 != v19 )
        {
          v76 = v21;
          v81 = v17;
          v88 = v20;
          srcc = v19;
          memmove((char *)v96 - (v18 - v19), v19, v18 - v19);
          v21 = v76;
          v17 = v81;
          LODWORD(v20) = v88;
          v19 = srcc;
        }
        if ( v21 )
        {
          v82 = v17;
          v89 = v20;
          srcd = (void *)v21;
          v22 = (char *)memmove(v19, v13, v21);
          v17 = v82;
          LODWORD(v20) = v89;
          v21 = (size_t)srcd;
          v19 = v22;
        }
        srca = &v19[v21];
      }
    }
    v90 = v17;
    sub_295AE70((_DWORD)dest, (_DWORD)v19, (_DWORD)srca, v20, v17, (_DWORD)v13, a7, a8);
    result = a7;
    v12 -= v90;
    if ( v12 <= a7 )
      result = v12;
    if ( v14 <= result )
    {
      a6 = v13;
      v11 = (char *)a3;
      v10 = (char *)v96;
      v9 = srca;
LABEL_22:
      v23 = v10 - v9;
      if ( v10 != v9 )
      {
        result = (__int64)memmove(a6, v9, v10 - v9);
        a6 = (char *)result;
      }
      v24 = &a6[v23];
      if ( a6 != &a6[v23] )
      {
        while ( 1 )
        {
          if ( v11 == v10 )
            goto LABEL_69;
          v25 = *(_DWORD *)(a8 + 24);
          v26 = *(_QWORD *)a6;
          v27 = *(_QWORD *)v10;
          v28 = *(_QWORD *)(a8 + 8);
          if ( !v25 )
            goto LABEL_41;
          v29 = v25 - 1;
          v30 = v29 & (((unsigned int)v27 >> 9) ^ ((unsigned int)v27 >> 4));
          v31 = (__int64 *)(v28 + 16LL * v30);
          v32 = *v31;
          if ( v27 == *v31 )
          {
LABEL_28:
            result = v31[1];
            if ( result )
            {
              v33 = *(_QWORD **)result;
              for ( result = 1; v33; result = (unsigned int)(result + 1) )
                v33 = (_QWORD *)*v33;
            }
          }
          else
          {
            v61 = 1;
            while ( v32 != -4096 )
            {
              v71 = v61 + 1;
              v30 = v29 & (v61 + v30);
              v31 = (__int64 *)(v28 + 16LL * v30);
              v32 = *v31;
              if ( v27 == *v31 )
                goto LABEL_28;
              v61 = v71;
            }
            result = 0;
          }
          v34 = v29 & (((unsigned int)v26 >> 9) ^ ((unsigned int)v26 >> 4));
          v35 = (__int64 *)(v28 + 16LL * v34);
          v36 = *v35;
          if ( v26 != *v35 )
            break;
LABEL_32:
          v37 = (_QWORD **)v35[1];
          if ( !v37 )
            goto LABEL_41;
          v38 = *v37;
          for ( i = 1; v38; ++i )
            v38 = (_QWORD *)*v38;
          if ( i <= (unsigned int)result )
            goto LABEL_41;
          v10 += 8;
LABEL_37:
          *(_QWORD *)v9 = v27;
          v9 += 8;
          if ( v24 == a6 )
            return result;
        }
        v40 = 1;
        while ( v36 != -4096 )
        {
          v34 = v29 & (v40 + v34);
          desta = v40 + 1;
          v35 = (__int64 *)(v28 + 16LL * v34);
          v36 = *v35;
          if ( v26 == *v35 )
            goto LABEL_32;
          v40 = desta;
        }
LABEL_41:
        a6 += 8;
        v27 = v26;
        goto LABEL_37;
      }
LABEL_69:
      if ( v24 == a6 )
        return result;
      v62 = a6;
      v63 = v9;
      v64 = v24 - a6;
      return (__int64)memmove(v63, v62, v64);
    }
    if ( v12 <= a7 )
      break;
    v15 = v96;
    dest = srca;
  }
  a6 = v13;
  v11 = (char *)a3;
  v10 = (char *)v96;
  v9 = srca;
LABEL_44:
  if ( v11 != v10 )
  {
    result = (__int64)memmove(a6, v10, v11 - v10);
    a6 = (char *)result;
  }
  v41 = &a6[v11 - v10];
  if ( v9 == v10 )
  {
    if ( a6 == v41 )
      return result;
    v64 = v11 - v10;
    v63 = v10;
LABEL_86:
    v62 = a6;
    return (__int64)memmove(v63, v62, v64);
  }
  if ( a6 == v41 )
    return result;
  v42 = v10 - 8;
  v43 = v41 - 8;
  v44 = v11 - 8;
  while ( 2 )
  {
    v45 = *(_DWORD *)(a8 + 24);
    v46 = *(_QWORD *)v42;
    v47 = *(_QWORD *)v43;
    v48 = *(_QWORD *)(a8 + 8);
    if ( !v45 )
      goto LABEL_64;
    v49 = v45 - 1;
    v50 = v49 & (((unsigned int)v47 >> 9) ^ ((unsigned int)v47 >> 4));
    v51 = (__int64 *)(v48 + 16LL * v50);
    v52 = *v51;
    if ( v47 == *v51 )
    {
LABEL_51:
      result = v51[1];
      if ( result )
      {
        v53 = *(_QWORD **)result;
        for ( result = 1; v53; result = (unsigned int)(result + 1) )
          v53 = (_QWORD *)*v53;
      }
    }
    else
    {
      v65 = 1;
      while ( v52 != -4096 )
      {
        v72 = v65 + 1;
        v50 = v49 & (v65 + v50);
        v51 = (__int64 *)(v48 + 16LL * v50);
        v52 = *v51;
        if ( v47 == *v51 )
          goto LABEL_51;
        v65 = v72;
      }
      result = 0;
    }
    v54 = v49 & (((unsigned int)v46 >> 9) ^ ((unsigned int)v46 >> 4));
    v55 = (__int64 *)(v48 + 16LL * v54);
    v56 = *v55;
    if ( v46 != *v55 )
    {
      v60 = 1;
      while ( v56 != -4096 )
      {
        v54 = v49 & (v60 + v54);
        destb = v60 + 1;
        v55 = (__int64 *)(v48 + 16LL * v54);
        v56 = *v55;
        if ( v46 == *v55 )
          goto LABEL_55;
        v60 = destb;
      }
      goto LABEL_64;
    }
LABEL_55:
    v57 = (_QWORD **)v55[1];
    if ( !v57 )
      goto LABEL_64;
    v58 = *v57;
    for ( j = 1; v58; ++j )
      v58 = (_QWORD *)*v58;
    if ( j <= (unsigned int)result )
    {
LABEL_64:
      *(_QWORD *)v44 = v47;
      if ( a6 == v43 )
        return result;
      v43 -= 8;
      goto LABEL_61;
    }
    *(_QWORD *)v44 = v46;
    if ( v42 != v9 )
    {
      v42 -= 8;
LABEL_61:
      v44 -= 8;
      continue;
    }
    break;
  }
  v69 = v43 + 8;
  if ( a6 != v69 )
  {
    v64 = v69 - a6;
    v63 = &v44[-(v69 - a6)];
    goto LABEL_86;
  }
  return result;
}
