// Function: sub_F7C410
// Address: 0xf7c410
//
__int64 __fastcall sub_F7C410(char *a1, char *a2, __int64 a3, __int64 a4, __int64 a5, char *a6, __int64 a7)
{
  __int64 result; // rax
  char *v8; // r14
  char *v10; // r12
  char *v11; // rbx
  __int64 v12; // r15
  __int64 v13; // r9
  __int64 v14; // rbx
  __int64 v15; // r11
  __int64 v16; // r13
  __int64 v17; // rax
  int v18; // r11d
  char *v19; // r9
  char *v20; // r12
  __int64 v21; // rcx
  char *v22; // r10
  size_t v23; // r10
  char *v24; // r15
  char *v25; // r9
  void *v26; // rcx
  __int64 v27; // rbx
  __int64 v28; // rdi
  size_t v29; // rdx
  char *v30; // r15
  char *v31; // rsi
  char *v32; // r15
  char *v33; // r9
  char *v34; // r13
  __int64 v35; // r12
  __int64 v36; // r14
  __int64 v37; // rdi
  const void *v38; // rsi
  char *v39; // rdi
  __int64 v40; // rax
  size_t v41; // r8
  char *v42; // rax
  char *v43; // rax
  int v44; // [rsp+8h] [rbp-88h]
  char *v45; // [rsp+8h] [rbp-88h]
  int v46; // [rsp+10h] [rbp-80h]
  int v47; // [rsp+10h] [rbp-80h]
  int v48; // [rsp+10h] [rbp-80h]
  int v49; // [rsp+10h] [rbp-80h]
  int v50; // [rsp+10h] [rbp-80h]
  int v51; // [rsp+10h] [rbp-80h]
  char *src; // [rsp+18h] [rbp-78h]
  void *srca; // [rsp+18h] [rbp-78h]
  void *srcb; // [rsp+18h] [rbp-78h]
  int srcc; // [rsp+18h] [rbp-78h]
  int srcd; // [rsp+18h] [rbp-78h]
  void *srce; // [rsp+18h] [rbp-78h]
  int srcf; // [rsp+18h] [rbp-78h]
  int srcg; // [rsp+18h] [rbp-78h]
  char *v61; // [rsp+20h] [rbp-70h]
  void *v62; // [rsp+28h] [rbp-68h]
  char *v63; // [rsp+28h] [rbp-68h]
  int v64; // [rsp+28h] [rbp-68h]
  void *v65; // [rsp+28h] [rbp-68h]
  char *v66; // [rsp+28h] [rbp-68h]
  void *v67; // [rsp+28h] [rbp-68h]
  char *v68; // [rsp+28h] [rbp-68h]
  char *v69; // [rsp+28h] [rbp-68h]
  void *v70; // [rsp+28h] [rbp-68h]
  int v71; // [rsp+28h] [rbp-68h]
  int v72; // [rsp+28h] [rbp-68h]
  int v73; // [rsp+28h] [rbp-68h]
  char *dest; // [rsp+30h] [rbp-60h]
  void *desta; // [rsp+30h] [rbp-60h]
  void *destb; // [rsp+30h] [rbp-60h]
  char *v77; // [rsp+38h] [rbp-58h]
  unsigned __int64 v78; // [rsp+50h] [rbp-40h]
  unsigned __int64 v79; // [rsp+50h] [rbp-40h]

  result = a5;
  v8 = a1;
  v10 = a2;
  v11 = (char *)a3;
  if ( a7 <= a5 )
    result = a7;
  if ( a4 <= result )
    goto LABEL_22;
  v12 = a5;
  if ( a7 >= a5 )
    goto LABEL_37;
  v13 = (__int64)a2;
  v14 = a4;
  v15 = (__int64)a1;
  dest = a6;
  while ( 1 )
  {
    if ( v14 > v12 )
    {
      v49 = v15;
      v69 = (char *)v13;
      v20 = (char *)(v15 + 8 * (v14 / 2));
      v40 = sub_F79F60(v13, a3, (__int64)v20);
      v19 = v69;
      v21 = v14 / 2;
      v77 = (char *)v40;
      v18 = v49;
      v16 = (v40 - (__int64)v69) >> 3;
    }
    else
    {
      src = (char *)v13;
      v62 = (void *)v15;
      v16 = v12 / 2;
      v77 = (char *)(v13 + 8 * (v12 / 2));
      v17 = sub_F7A030(v15, v13, (__int64)v77);
      v18 = (int)v62;
      v19 = src;
      v20 = (char *)v17;
      v21 = (v17 - (__int64)v62) >> 3;
    }
    v14 -= v21;
    if ( v14 <= v16 || v16 > a7 )
    {
      if ( v14 > a7 )
      {
        srcg = v18;
        v73 = v21;
        v43 = sub_F7A510(v20, v19, v77);
        v18 = srcg;
        LODWORD(v21) = v73;
        v22 = v43;
      }
      else
      {
        v22 = v77;
        if ( v14 )
        {
          v41 = v19 - v20;
          if ( v19 != v20 )
          {
            v50 = v18;
            v45 = v19;
            srcd = v21;
            v70 = (void *)(v19 - v20);
            memmove(dest, v20, v19 - v20);
            v19 = v45;
            v18 = v50;
            LODWORD(v21) = srcd;
            v41 = (size_t)v70;
          }
          if ( v19 != v77 )
          {
            v51 = v18;
            srce = (void *)v41;
            v71 = v21;
            memmove(v20, v19, v77 - v19);
            v18 = v51;
            v41 = (size_t)srce;
            LODWORD(v21) = v71;
          }
          v22 = &v77[-v41];
          if ( v41 )
          {
            srcf = v18;
            v72 = v21;
            v42 = (char *)memmove(&v77[-v41], dest, v41);
            LODWORD(v21) = v72;
            v18 = srcf;
            v22 = v42;
          }
        }
      }
    }
    else
    {
      v22 = v20;
      if ( v16 )
      {
        v23 = v77 - v19;
        if ( v19 != v77 )
        {
          v46 = v21;
          v44 = v18;
          srca = (void *)(v77 - v19);
          v63 = v19;
          memmove(dest, v19, v77 - v19);
          v18 = v44;
          LODWORD(v21) = v46;
          v23 = (size_t)srca;
          v19 = v63;
        }
        if ( v19 != v20 )
        {
          v47 = v18;
          srcb = (void *)v23;
          v64 = v21;
          memmove(&v77[-(v19 - v20)], v20, v19 - v20);
          v18 = v47;
          v23 = (size_t)srcb;
          LODWORD(v21) = v64;
        }
        if ( v23 )
        {
          v48 = v18;
          srcc = v21;
          v65 = (void *)v23;
          memmove(v20, dest, v23);
          v18 = v48;
          LODWORD(v21) = srcc;
          v23 = (size_t)v65;
        }
        v22 = &v20[v23];
      }
    }
    v12 -= v16;
    v66 = v22;
    sub_F7C410(v18, (_DWORD)v20, (_DWORD)v22, v21, v16, (_DWORD)dest, a7);
    result = a7;
    if ( v12 <= a7 )
      result = v12;
    if ( v14 <= result )
    {
      v11 = (char *)a3;
      a6 = dest;
      v8 = v66;
      v10 = v77;
LABEL_22:
      if ( v8 != v10 )
        result = (__int64)memmove(a6, v8, v10 - v8);
      v24 = &a6[v10 - v8];
      if ( v11 != v10 )
      {
        if ( v24 == a6 )
          return result;
        v25 = v11;
        do
        {
          v26 = *(void **)v10;
          v27 = *(_QWORD *)a6;
          v28 = *(_QWORD *)(*(_QWORD *)a6 + 8LL);
          result = *(unsigned __int8 *)(v28 + 8);
          if ( *(_BYTE *)(*(_QWORD *)(*(_QWORD *)v10 + 8LL) + 8LL) == 12 )
          {
            if ( (_BYTE)result != 12
              || (v61 = v25,
                  desta = *(void **)v10,
                  v67 = *(void **)(*(_QWORD *)v10 + 8LL),
                  v78 = sub_BCAE30(v28),
                  result = sub_BCAE30((__int64)v67),
                  v26 = desta,
                  v25 = v61,
                  v78 >= result) )
            {
LABEL_28:
              a6 += 8;
              *(_QWORD *)v8 = v27;
              v8 += 8;
              if ( v24 == a6 )
                return result;
              continue;
            }
          }
          else if ( (_BYTE)result != 12 )
          {
            goto LABEL_28;
          }
          v10 += 8;
          v8 += 8;
          *((_QWORD *)v8 - 1) = v26;
          if ( v24 == a6 )
            return result;
        }
        while ( v25 != v10 );
      }
      if ( v24 == a6 )
        return result;
      v38 = a6;
      v39 = v8;
      v29 = v24 - a6;
      return (__int64)memmove(v39, v38, v29);
    }
    if ( v12 <= a7 )
      break;
    v13 = (__int64)v77;
    v15 = (__int64)v66;
  }
  v11 = (char *)a3;
  a6 = dest;
  v8 = v66;
  v10 = v77;
LABEL_37:
  v29 = v11 - v10;
  if ( v11 != v10 )
  {
    result = (__int64)memmove(a6, v10, v29);
    v29 = v11 - v10;
  }
  v30 = &a6[v29];
  if ( v10 == v8 )
  {
    if ( a6 == v30 )
      return result;
LABEL_65:
    v38 = a6;
    v39 = &v11[-v29];
    return (__int64)memmove(v39, v38, v29);
  }
  if ( a6 == v30 )
    return result;
  v31 = a6;
  v32 = v30 - 8;
  v11 -= 8;
  v33 = v8;
  v34 = v10 - 8;
  while ( 2 )
  {
    v35 = *(_QWORD *)v32;
    v36 = *(_QWORD *)v34;
    v37 = *(_QWORD *)(*(_QWORD *)v34 + 8LL);
    result = *(unsigned __int8 *)(v37 + 8);
    if ( *(_BYTE *)(*(_QWORD *)(*(_QWORD *)v32 + 8LL) + 8LL) != 12 )
    {
      if ( (_BYTE)result == 12 )
        goto LABEL_49;
      goto LABEL_43;
    }
    if ( (_BYTE)result != 12
      || (v68 = v33,
          destb = *(void **)(*(_QWORD *)v32 + 8LL),
          v79 = sub_BCAE30(v37),
          result = sub_BCAE30((__int64)destb),
          v33 = v68,
          v79 >= result) )
    {
LABEL_43:
      *(_QWORD *)v11 = v35;
      if ( v31 == v32 )
        return result;
      v32 -= 8;
      goto LABEL_45;
    }
LABEL_49:
    *(_QWORD *)v11 = v36;
    if ( v34 != v33 )
    {
      v34 -= 8;
LABEL_45:
      v11 -= 8;
      continue;
    }
    break;
  }
  a6 = v31;
  if ( v31 != v32 + 8 )
  {
    v29 = v32 + 8 - v31;
    goto LABEL_65;
  }
  return result;
}
