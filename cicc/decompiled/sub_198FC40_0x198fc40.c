// Function: sub_198FC40
// Address: 0x198fc40
//
void **__fastcall sub_198FC40(
        __int64 *a1,
        __int64 *a2,
        char *a3,
        __int64 a4,
        __int64 a5,
        void **a6,
        __int64 a7,
        __int64 *a8)
{
  __int64 v8; // rax
  __int64 *v9; // r15
  __int64 *v10; // r12
  __int64 v12; // r13
  __int64 *v13; // r9
  __int64 v14; // r12
  __int64 v15; // rbx
  __int64 *v16; // rax
  char *v17; // r9
  char *v18; // r10
  __int64 v19; // r11
  size_t v20; // rcx
  char *v21; // rax
  __int64 v22; // rax
  signed __int64 v23; // r13
  __int64 *v24; // rax
  __int64 *v25; // r15
  __int64 *v26; // r12
  void **result; // rax
  __int64 v28; // r13
  __int64 *v29; // rbx
  __int64 *i; // r14
  __int64 v31; // r13
  unsigned __int64 v32; // r12
  void **v33; // rsi
  __int64 *v34; // rdi
  size_t v35; // rdx
  __int64 *v36; // rax
  size_t v37; // r8
  unsigned int v38; // eax
  __int64 *v39; // r15
  char *v40; // rax
  char *v41; // [rsp+0h] [rbp-70h]
  char *v42; // [rsp+0h] [rbp-70h]
  int v43; // [rsp+8h] [rbp-68h]
  size_t v44; // [rsp+8h] [rbp-68h]
  int v45; // [rsp+8h] [rbp-68h]
  int v46; // [rsp+8h] [rbp-68h]
  int v47; // [rsp+8h] [rbp-68h]
  size_t v48; // [rsp+10h] [rbp-60h]
  int v49; // [rsp+10h] [rbp-60h]
  int v50; // [rsp+10h] [rbp-60h]
  size_t v51; // [rsp+10h] [rbp-60h]
  size_t v52; // [rsp+10h] [rbp-60h]
  int v53; // [rsp+10h] [rbp-60h]
  int v54; // [rsp+10h] [rbp-60h]
  void **dest; // [rsp+18h] [rbp-58h]
  __int64 *v56; // [rsp+20h] [rbp-50h]
  __int64 *v57; // [rsp+28h] [rbp-48h]
  void **v58; // [rsp+28h] [rbp-48h]
  __int64 *v59; // [rsp+28h] [rbp-48h]
  __int64 *src; // [rsp+30h] [rbp-40h]
  char *srca; // [rsp+30h] [rbp-40h]
  char *srcc; // [rsp+30h] [rbp-40h]
  char *srcd; // [rsp+30h] [rbp-40h]
  void *srce; // [rsp+30h] [rbp-40h]
  void *srcf; // [rsp+30h] [rbp-40h]
  __int64 *srcb; // [rsp+30h] [rbp-40h]
  char *srcg; // [rsp+30h] [rbp-40h]
  int srch; // [rsp+30h] [rbp-40h]

  v8 = a5;
  v9 = a2;
  v10 = a1;
  if ( a7 <= a5 )
    v8 = a7;
  if ( a4 <= v8 )
  {
LABEL_22:
    v23 = (char *)v9 - (char *)v10;
    if ( v9 != v10 )
      memmove(a6, v10, (char *)v9 - (char *)v10);
    v58 = (void **)((char *)a6 + v23);
    if ( a6 != (void **)((char *)a6 + v23) && a3 != (char *)v9 )
    {
      v24 = v9;
      v25 = v10;
      v26 = v24;
      do
      {
        v28 = *v26;
        srcf = (void *)sub_1368AA0(a8, (__int64)*a6);
        if ( (unsigned __int64)srcf > sub_1368AA0(a8, v28) )
        {
          result = (void **)*v26;
          ++v25;
          ++v26;
          *(v25 - 1) = (__int64)result;
          if ( v58 == a6 )
            return result;
        }
        else
        {
          result = (void **)*a6;
          ++v25;
          ++a6;
          *(v25 - 1) = (__int64)result;
          if ( v58 == a6 )
            return result;
        }
      }
      while ( a3 != (char *)v26 );
      v10 = v25;
    }
    result = v58;
    if ( v58 != a6 )
    {
      v33 = a6;
      v34 = v10;
      v35 = (char *)v58 - (char *)a6;
      return (void **)memmove(v34, v33, v35);
    }
  }
  else
  {
    v12 = a5;
    if ( a7 < a5 )
    {
      v57 = a1;
      v13 = a2;
      v14 = a4;
      dest = a6;
      while ( 1 )
      {
        src = v13;
        if ( v14 > v12 )
        {
          v36 = sub_198EA20(v13, (__int64)a3, &v57[v14 / 2], a8);
          v17 = (char *)src;
          v18 = (char *)&v57[v14 / 2];
          v56 = v36;
          v19 = v14 / 2;
          v15 = v36 - src;
        }
        else
        {
          v15 = v12 / 2;
          v56 = &v13[v12 / 2];
          v16 = sub_198E980(v57, (__int64)v13, v56, a8);
          v17 = (char *)src;
          v18 = (char *)v16;
          v19 = v16 - v57;
        }
        v14 -= v19;
        if ( v14 <= v15 || v15 > a7 )
        {
          if ( v14 > a7 )
          {
            v47 = v19;
            v54 = (int)v18;
            v40 = sub_198F840(v18, v17, (char *)v56);
            LODWORD(v19) = v47;
            LODWORD(v18) = v54;
            srca = v40;
          }
          else
          {
            srca = (char *)v56;
            if ( v14 )
            {
              v37 = v17 - v18;
              if ( v17 != v18 )
              {
                v42 = v17;
                v45 = v19;
                v51 = v17 - v18;
                srcg = v18;
                memmove(dest, v18, v17 - v18);
                v17 = v42;
                LODWORD(v19) = v45;
                v37 = v51;
                v18 = srcg;
              }
              if ( v17 != (char *)v56 )
              {
                v52 = v37;
                srch = v19;
                v38 = (unsigned int)memmove(v18, v17, (char *)v56 - v17);
                v37 = v52;
                LODWORD(v19) = srch;
                LODWORD(v18) = v38;
              }
              srca = (char *)v56 - v37;
              if ( v37 )
              {
                v46 = (int)v18;
                v53 = v19;
                memmove((char *)v56 - v37, dest, v37);
                LODWORD(v19) = v53;
                LODWORD(v18) = v46;
              }
            }
          }
        }
        else
        {
          srca = v18;
          if ( v15 )
          {
            v20 = (char *)v56 - v17;
            if ( v17 != (char *)v56 )
            {
              v41 = v18;
              v43 = v19;
              v48 = (char *)v56 - v17;
              srcc = v17;
              memmove(dest, v17, (char *)v56 - v17);
              v18 = v41;
              LODWORD(v19) = v43;
              v20 = v48;
              v17 = srcc;
            }
            if ( v17 != v18 )
            {
              v44 = v20;
              v49 = v19;
              srcd = v18;
              memmove((char *)v56 - (v17 - v18), v18, v17 - v18);
              v20 = v44;
              LODWORD(v19) = v49;
              v18 = srcd;
            }
            if ( v20 )
            {
              v50 = v19;
              srce = (void *)v20;
              v21 = (char *)memmove(v18, dest, v20);
              LODWORD(v19) = v50;
              v20 = (size_t)srce;
              v18 = v21;
            }
            srca = &v18[v20];
          }
        }
        v12 -= v15;
        sub_198FC40((_DWORD)v57, (_DWORD)v18, (_DWORD)srca, v19, v15, (_DWORD)dest, a7, (__int64)a8);
        v22 = a7;
        if ( v12 <= a7 )
          v22 = v12;
        if ( v14 <= v22 )
        {
          a6 = dest;
          v9 = v56;
          v10 = (__int64 *)srca;
          goto LABEL_22;
        }
        if ( v12 <= a7 )
          break;
        v13 = v56;
        v57 = (__int64 *)srca;
      }
      a6 = dest;
      v9 = v56;
      v10 = (__int64 *)srca;
    }
    if ( a3 != (char *)v9 )
      memmove(a6, v9, a3 - (char *)v9);
    result = (void **)((char *)a6 + a3 - (char *)v9);
    if ( v9 == v10 )
    {
      if ( a6 != result )
      {
        v35 = a3 - (char *)v9;
        v34 = v9;
        goto LABEL_59;
      }
    }
    else if ( a6 != result )
    {
      v59 = v10;
      srcb = (__int64 *)a6;
      v29 = (__int64 *)(result - 1);
      for ( i = v9 - 1; ; --i )
      {
        while ( 1 )
        {
          v31 = *v29;
          v32 = sub_1368AA0(a8, *i);
          a3 -= 8;
          if ( v32 > sub_1368AA0(a8, v31) )
            break;
          result = (void **)*v29;
          *(_QWORD *)a3 = *v29;
          if ( srcb == v29 )
            return result;
          --v29;
        }
        result = (void **)*i;
        *(_QWORD *)a3 = *i;
        if ( i == v59 )
          break;
      }
      v39 = v29;
      a6 = (void **)srcb;
      if ( srcb != v39 + 1 )
      {
        v35 = (char *)(v39 + 1) - (char *)srcb;
        v34 = (__int64 *)&a3[-v35];
LABEL_59:
        v33 = a6;
        return (void **)memmove(v34, v33, v35);
      }
    }
  }
  return result;
}
