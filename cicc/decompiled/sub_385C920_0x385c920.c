// Function: sub_385C920
// Address: 0x385c920
//
unsigned int *__fastcall sub_385C920(
        char *a1,
        unsigned int *a2,
        __int64 a3,
        __int64 a4,
        __int64 a5,
        unsigned int *a6,
        unsigned int *a7,
        _QWORD *a8)
{
  unsigned int *result; // rax
  char *v9; // r13
  char *v10; // r12
  char *v11; // rbx
  _QWORD *v12; // r14
  __int64 v13; // r15
  __int64 v14; // rbx
  unsigned int *i; // r9
  __int64 v16; // r14
  unsigned int *v17; // rax
  char *v18; // r9
  unsigned int *v19; // r12
  __int64 v20; // r11
  char *v21; // r10
  size_t v22; // r10
  signed __int64 v23; // r15
  unsigned int *v24; // r8
  __int64 v25; // rdi
  __int64 v26; // rcx
  unsigned int *v27; // rsi
  unsigned int *v28; // r12
  __int64 v29; // r8
  __int64 v30; // rdi
  unsigned int *v31; // rsi
  char *v32; // rdi
  size_t v33; // rdx
  unsigned int *v34; // rax
  size_t v35; // r8
  char *v36; // rax
  char *v37; // rax
  int v38; // [rsp+0h] [rbp-70h]
  char *v39; // [rsp+0h] [rbp-70h]
  size_t v40; // [rsp+8h] [rbp-68h]
  size_t v41; // [rsp+8h] [rbp-68h]
  int v42; // [rsp+8h] [rbp-68h]
  int v43; // [rsp+8h] [rbp-68h]
  size_t v44; // [rsp+8h] [rbp-68h]
  unsigned int *src; // [rsp+18h] [rbp-58h]
  char *srcb; // [rsp+18h] [rbp-58h]
  int srcc; // [rsp+18h] [rbp-58h]
  void *srcd; // [rsp+18h] [rbp-58h]
  char *srca; // [rsp+18h] [rbp-58h]
  void *srce; // [rsp+18h] [rbp-58h]
  int srcf; // [rsp+18h] [rbp-58h]
  int srcg; // [rsp+18h] [rbp-58h]
  int srch; // [rsp+18h] [rbp-58h]
  unsigned int *dest; // [rsp+20h] [rbp-50h]
  unsigned int *v56; // [rsp+28h] [rbp-48h]
  unsigned int *v57; // [rsp+30h] [rbp-40h]

  result = (unsigned int *)a5;
  v9 = a1;
  v10 = (char *)a2;
  v11 = (char *)a3;
  v12 = a8;
  if ( (__int64)a7 <= a5 )
    result = a7;
  if ( a4 <= (__int64)result )
  {
LABEL_22:
    v23 = v10 - v9;
    if ( v10 != v9 )
    {
      result = (unsigned int *)memmove(a6, v9, v10 - v9);
      a6 = result;
    }
    v24 = (unsigned int *)((char *)a6 + v23);
    if ( a6 != (unsigned int *)((char *)a6 + v23) )
    {
      while ( v11 != v10 )
      {
        v25 = *a6;
        v26 = *(unsigned int *)v10;
        result = *(unsigned int **)(*v12 + 16 * v25);
        if ( *(_QWORD *)(*v12 + 16 * v26) < (__int64)result )
        {
          *(_DWORD *)v9 = v26;
          v10 += 4;
          v9 += 4;
          if ( v24 == a6 )
            return result;
        }
        else
        {
          ++a6;
          v9 += 4;
          *((_DWORD *)v9 - 1) = v25;
          if ( v24 == a6 )
            return result;
        }
      }
    }
    if ( v24 != a6 )
    {
      v31 = a6;
      v32 = v9;
      v33 = (char *)v24 - (char *)a6;
      return (unsigned int *)memmove(v32, v31, v33);
    }
  }
  else
  {
    v13 = a5;
    if ( (__int64)a7 < a5 )
    {
      v14 = a4;
      v57 = (unsigned int *)a1;
      dest = a6;
      for ( i = a2; ; i = v56 )
      {
        src = i;
        if ( v14 > v13 )
        {
          v19 = &v57[v14 / 2];
          v34 = sub_385BC10(i, a3, v19, a8);
          v18 = (char *)src;
          v20 = v14 / 2;
          v56 = v34;
          v16 = v34 - src;
        }
        else
        {
          v16 = v13 / 2;
          v56 = &i[v13 / 2];
          v17 = sub_385BBB0(v57, (__int64)i, v56, a8);
          v18 = (char *)src;
          v19 = v17;
          v20 = v17 - v57;
        }
        v14 -= v20;
        if ( v14 <= v16 || v16 > (__int64)a7 )
        {
          if ( v14 > (__int64)a7 )
          {
            srch = v20;
            v37 = sub_385C1E0((char *)v19, v18, (char *)v56);
            LODWORD(v20) = srch;
            v21 = v37;
          }
          else
          {
            v21 = (char *)v56;
            if ( v14 )
            {
              v35 = v18 - (char *)v19;
              if ( v18 != (char *)v19 )
              {
                v39 = v18;
                v43 = v20;
                srce = (void *)(v18 - (char *)v19);
                memmove(dest, v19, v18 - (char *)v19);
                v18 = v39;
                LODWORD(v20) = v43;
                v35 = (size_t)srce;
              }
              if ( v18 != (char *)v56 )
              {
                v44 = v35;
                srcf = v20;
                memmove(v19, v18, (char *)v56 - v18);
                v35 = v44;
                LODWORD(v20) = srcf;
              }
              v21 = (char *)v56 - v35;
              if ( v35 )
              {
                srcg = v20;
                v36 = (char *)memmove((char *)v56 - v35, dest, v35);
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
            v22 = (char *)v56 - v18;
            if ( v18 != (char *)v56 )
            {
              v38 = v20;
              v40 = (char *)v56 - v18;
              srcb = v18;
              memmove(dest, v18, (char *)v56 - v18);
              LODWORD(v20) = v38;
              v22 = v40;
              v18 = srcb;
            }
            if ( v18 != (char *)v19 )
            {
              v41 = v22;
              srcc = v20;
              memmove((char *)v56 - (v18 - (char *)v19), v19, v18 - (char *)v19);
              v22 = v41;
              LODWORD(v20) = srcc;
            }
            if ( v22 )
            {
              v42 = v20;
              srcd = (void *)v22;
              memmove(v19, dest, v22);
              LODWORD(v20) = v42;
              v22 = (size_t)srcd;
            }
            v21 = (char *)v19 + v22;
          }
        }
        v13 -= v16;
        srca = v21;
        sub_385C920((_DWORD)v57, (_DWORD)v19, (_DWORD)v21, v20, v16, (_DWORD)dest, (__int64)a7, (__int64)a8);
        result = a7;
        if ( v13 <= (__int64)a7 )
          result = (unsigned int *)v13;
        if ( v14 <= (__int64)result )
        {
          v12 = a8;
          v11 = (char *)a3;
          v9 = srca;
          a6 = dest;
          v10 = (char *)v56;
          goto LABEL_22;
        }
        if ( v13 <= (__int64)a7 )
          break;
        v57 = (unsigned int *)srca;
      }
      v12 = a8;
      v11 = (char *)a3;
      v9 = srca;
      a6 = dest;
      v10 = (char *)v56;
    }
    if ( v11 != v10 )
    {
      result = (unsigned int *)memmove(a6, v10, v11 - v10);
      a6 = result;
    }
    v27 = (unsigned int *)((char *)a6 + v11 - v10);
    if ( v10 == v9 )
    {
      if ( a6 != v27 )
      {
        v33 = v11 - v10;
        v32 = v10;
LABEL_58:
        v31 = a6;
        return (unsigned int *)memmove(v32, v31, v33);
      }
    }
    else if ( a6 != v27 )
    {
      v28 = (unsigned int *)(v10 - 4);
LABEL_38:
      --v27;
      while ( 1 )
      {
        v11 -= 4;
        v29 = *v28;
        v30 = *v27;
        result = *(unsigned int **)(*v12 + 16 * v29);
        if ( *(_QWORD *)(*v12 + 16 * v30) >= (__int64)result )
        {
          *(_DWORD *)v11 = v30;
          if ( a6 != v27 )
            goto LABEL_38;
          return result;
        }
        *(_DWORD *)v11 = v29;
        if ( v28 == (unsigned int *)v9 )
          break;
        --v28;
      }
      if ( a6 == v27 + 1 )
        return result;
      v33 = (char *)(v27 + 1) - (char *)a6;
      v32 = &v11[-v33];
      goto LABEL_58;
    }
  }
  return result;
}
