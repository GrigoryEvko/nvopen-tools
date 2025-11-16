// Function: sub_1F22140
// Address: 0x1f22140
//
__int64 __fastcall sub_1F22140(
        char *a1,
        char *a2,
        __int64 a3,
        __int64 a4,
        __int64 a5,
        unsigned int *a6,
        __int64 a7,
        __int64 a8)
{
  __int64 result; // rax
  char *v9; // r13
  char *v10; // r12
  unsigned int *v11; // rbx
  __int64 v12; // r14
  __int64 v13; // r15
  __int64 v14; // rbx
  int *i; // r9
  __int64 v16; // r14
  int *v17; // rax
  char *v18; // r9
  int *v19; // r12
  __int64 v20; // r11
  char *v21; // r10
  size_t v22; // r10
  signed __int64 v23; // r15
  char *v24; // r8
  unsigned int v25; // edx
  unsigned int *v26; // rdx
  unsigned int *v27; // r12
  unsigned int *v28; // rdx
  unsigned int *v29; // rbx
  unsigned int v30; // ecx
  unsigned int *v31; // rsi
  char *v32; // rdi
  size_t v33; // rdx
  int *v34; // rax
  size_t v35; // r8
  char *v36; // rax
  unsigned int *v37; // rdx
  char *v38; // rax
  int v39; // [rsp+0h] [rbp-70h]
  char *v40; // [rsp+0h] [rbp-70h]
  size_t v41; // [rsp+8h] [rbp-68h]
  size_t v42; // [rsp+8h] [rbp-68h]
  int v43; // [rsp+8h] [rbp-68h]
  int v44; // [rsp+8h] [rbp-68h]
  size_t v45; // [rsp+8h] [rbp-68h]
  char *src; // [rsp+18h] [rbp-58h]
  char *srcb; // [rsp+18h] [rbp-58h]
  int srcc; // [rsp+18h] [rbp-58h]
  void *srcd; // [rsp+18h] [rbp-58h]
  char *srca; // [rsp+18h] [rbp-58h]
  void *srce; // [rsp+18h] [rbp-58h]
  int srcf; // [rsp+18h] [rbp-58h]
  int srcg; // [rsp+18h] [rbp-58h]
  int srch; // [rsp+18h] [rbp-58h]
  unsigned int *dest; // [rsp+20h] [rbp-50h]
  int *v57; // [rsp+28h] [rbp-48h]
  char *v58; // [rsp+30h] [rbp-40h]

  result = a5;
  v9 = a1;
  v10 = a2;
  v11 = (unsigned int *)a3;
  v12 = a8;
  if ( a7 <= a5 )
    result = a7;
  if ( a4 <= result )
    goto LABEL_22;
  v13 = a5;
  if ( a7 >= a5 )
    goto LABEL_35;
  v14 = a4;
  v58 = a1;
  dest = a6;
  for ( i = (int *)a2; ; i = v57 )
  {
    src = (char *)i;
    if ( v14 > v13 )
    {
      v19 = (int *)&v58[4 * (v14 / 2)];
      v34 = sub_1F20B50(i, a3, v19, a8);
      v18 = src;
      v20 = v14 / 2;
      v57 = v34;
      v16 = ((char *)v34 - src) >> 2;
    }
    else
    {
      v16 = v13 / 2;
      v57 = &i[v13 / 2];
      v17 = sub_1F20BE0(v58, (__int64)i, v57, a8);
      v18 = src;
      v19 = v17;
      v20 = ((char *)v17 - v58) >> 2;
    }
    v14 -= v20;
    if ( v14 <= v16 || v16 > a7 )
    {
      if ( v14 > a7 )
      {
        srch = v20;
        v38 = sub_1F21920((char *)v19, v18, (char *)v57);
        LODWORD(v20) = srch;
        v21 = v38;
      }
      else
      {
        v21 = (char *)v57;
        if ( v14 )
        {
          v35 = v18 - (char *)v19;
          if ( v18 != (char *)v19 )
          {
            v40 = v18;
            v44 = v20;
            srce = (void *)(v18 - (char *)v19);
            memmove(dest, v19, v18 - (char *)v19);
            v18 = v40;
            LODWORD(v20) = v44;
            v35 = (size_t)srce;
          }
          if ( v18 != (char *)v57 )
          {
            v45 = v35;
            srcf = v20;
            memmove(v19, v18, (char *)v57 - v18);
            v35 = v45;
            LODWORD(v20) = srcf;
          }
          v21 = (char *)v57 - v35;
          if ( v35 )
          {
            srcg = v20;
            v36 = (char *)memmove((char *)v57 - v35, dest, v35);
            LODWORD(v20) = srcg;
            v21 = v36;
          }
        }
      }
    }
    else
    {
      v21 = (char *)v19;
      if ( v16 )
      {
        v22 = (char *)v57 - v18;
        if ( v18 != (char *)v57 )
        {
          v39 = v20;
          v41 = (char *)v57 - v18;
          srcb = v18;
          memmove(dest, v18, (char *)v57 - v18);
          LODWORD(v20) = v39;
          v22 = v41;
          v18 = srcb;
        }
        if ( v18 != (char *)v19 )
        {
          v42 = v22;
          srcc = v20;
          memmove((char *)v57 - (v18 - (char *)v19), v19, v18 - (char *)v19);
          v22 = v42;
          LODWORD(v20) = srcc;
        }
        if ( v22 )
        {
          v43 = v20;
          srcd = (void *)v22;
          memmove(v19, dest, v22);
          LODWORD(v20) = v43;
          v22 = (size_t)srcd;
        }
        v21 = (char *)v19 + v22;
      }
    }
    v13 -= v16;
    srca = v21;
    sub_1F22140((_DWORD)v58, (_DWORD)v19, (_DWORD)v21, v20, v16, (_DWORD)dest, a7, a8);
    result = a7;
    if ( v13 <= a7 )
      result = v13;
    if ( v14 <= result )
    {
      v12 = a8;
      v11 = (unsigned int *)a3;
      v9 = srca;
      a6 = dest;
      v10 = (char *)v57;
LABEL_22:
      v23 = v10 - v9;
      if ( v10 != v9 )
      {
        result = (__int64)memmove(a6, v9, v10 - v9);
        a6 = (unsigned int *)result;
      }
      v24 = (char *)a6 + v23;
      if ( a6 != (unsigned int *)((char *)a6 + v23) )
      {
        while ( v11 != (unsigned int *)v10 )
        {
          result = *(unsigned int *)v10;
          v25 = *a6;
          if ( (_DWORD)result != -1
            && (v25 == -1
             || *(_QWORD *)(*(_QWORD *)(*(_QWORD *)(v12 + 232) + 8LL)
                          + 40LL * (unsigned int)(result + *(_DWORD *)(*(_QWORD *)(v12 + 232) + 32LL))
                          + 8) > *(_QWORD *)(*(_QWORD *)(*(_QWORD *)(v12 + 232) + 8LL)
                                           + 40LL * (v25 + *(_DWORD *)(*(_QWORD *)(v12 + 232) + 32LL))
                                           + 8)) )
          {
            *(_DWORD *)v9 = result;
            v10 += 4;
            v9 += 4;
            if ( v24 == (char *)a6 )
              return result;
          }
          else
          {
            result = v25;
            ++a6;
            v9 += 4;
            *((_DWORD *)v9 - 1) = v25;
            if ( v24 == (char *)a6 )
              return result;
          }
        }
      }
      if ( v24 == (char *)a6 )
        return result;
      v31 = a6;
      v32 = v9;
      v33 = v24 - (char *)a6;
      return (__int64)memmove(v32, v31, v33);
    }
    if ( v13 <= a7 )
      break;
    v58 = srca;
  }
  v12 = a8;
  v11 = (unsigned int *)a3;
  v9 = srca;
  a6 = dest;
  v10 = (char *)v57;
LABEL_35:
  if ( v11 != (unsigned int *)v10 )
  {
    result = (__int64)memmove(a6, v10, (char *)v11 - v10);
    a6 = (unsigned int *)result;
  }
  v26 = (unsigned int *)((char *)a6 + (char *)v11 - v10);
  if ( v10 == v9 )
  {
    if ( a6 == v26 )
      return result;
    v33 = (char *)v11 - v10;
    v32 = v10;
LABEL_62:
    v31 = a6;
    return (__int64)memmove(v32, v31, v33);
  }
  if ( a6 == v26 )
    return result;
  v27 = (unsigned int *)(v10 - 4);
  v28 = v26 - 1;
  v29 = v11 - 1;
  while ( 2 )
  {
    result = *v28;
    if ( (_DWORD)result == -1
      || (v30 = *v27, *v27 != -1)
      && *(_QWORD *)(*(_QWORD *)(*(_QWORD *)(v12 + 232) + 8LL)
                   + 40LL * (unsigned int)(result + *(_DWORD *)(*(_QWORD *)(v12 + 232) + 32LL))
                   + 8) <= *(_QWORD *)(*(_QWORD *)(*(_QWORD *)(v12 + 232) + 8LL)
                                     + 40LL * (v30 + *(_DWORD *)(*(_QWORD *)(v12 + 232) + 32LL))
                                     + 8) )
    {
      *v29 = result;
      if ( a6 == v28 )
        return result;
      --v28;
      goto LABEL_42;
    }
    *v29 = v30;
    if ( v27 != (unsigned int *)v9 )
    {
      --v27;
LABEL_42:
      --v29;
      continue;
    }
    break;
  }
  v37 = v28 + 1;
  if ( a6 != v37 )
  {
    v33 = (char *)v37 - (char *)a6;
    v32 = (char *)v29 - v33;
    goto LABEL_62;
  }
  return result;
}
