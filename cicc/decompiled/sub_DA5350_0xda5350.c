// Function: sub_DA5350
// Address: 0xda5350
//
void *__fastcall sub_DA5350(
        unsigned __int64 *a1,
        unsigned __int64 *a2,
        __int64 a3,
        __int64 a4,
        __int64 a5,
        unsigned __int64 *a6,
        void *a7,
        _QWORD **a8)
{
  void *result; // rax
  unsigned __int64 *v9; // r14
  unsigned __int64 *v11; // r12
  __int64 v12; // r15
  unsigned __int64 *v13; // r9
  __int64 v14; // r12
  __int64 v15; // r13
  unsigned __int64 *v16; // rax
  char *v17; // r9
  char *v18; // r10
  __int64 v19; // r11
  size_t v20; // rcx
  char *v21; // rax
  unsigned __int64 *v22; // r15
  size_t v23; // rdx
  unsigned __int64 *v24; // r15
  unsigned __int64 *v25; // r15
  unsigned __int64 *v26; // r14
  _QWORD *v27; // rbx
  unsigned __int64 *v28; // rsi
  unsigned __int64 *v29; // rdi
  unsigned __int64 *v30; // rax
  size_t v31; // r8
  unsigned int v32; // eax
  _QWORD *v33; // rdi
  char *v34; // rax
  char *v35; // [rsp+0h] [rbp-80h]
  char *v36; // [rsp+0h] [rbp-80h]
  int v37; // [rsp+8h] [rbp-78h]
  size_t v38; // [rsp+8h] [rbp-78h]
  int v39; // [rsp+8h] [rbp-78h]
  int v40; // [rsp+8h] [rbp-78h]
  int v41; // [rsp+8h] [rbp-78h]
  size_t v42; // [rsp+10h] [rbp-70h]
  int v43; // [rsp+10h] [rbp-70h]
  int v44; // [rsp+10h] [rbp-70h]
  size_t v45; // [rsp+10h] [rbp-70h]
  size_t v46; // [rsp+10h] [rbp-70h]
  int v47; // [rsp+10h] [rbp-70h]
  int v48; // [rsp+10h] [rbp-70h]
  unsigned __int64 *dest; // [rsp+18h] [rbp-68h]
  unsigned __int64 *v50; // [rsp+20h] [rbp-60h]
  unsigned __int64 *v51; // [rsp+28h] [rbp-58h]
  unsigned __int64 *src; // [rsp+30h] [rbp-50h]
  char *srca; // [rsp+30h] [rbp-50h]
  char *srcc; // [rsp+30h] [rbp-50h]
  char *srcd; // [rsp+30h] [rbp-50h]
  void *srce; // [rsp+30h] [rbp-50h]
  unsigned __int64 *srcb; // [rsp+30h] [rbp-50h]
  char *srcf; // [rsp+30h] [rbp-50h]
  int srcg; // [rsp+30h] [rbp-50h]
  __int64 v61; // [rsp+48h] [rbp-38h]
  __int64 v62; // [rsp+48h] [rbp-38h]

  result = (void *)a5;
  v9 = a1;
  v11 = a2;
  if ( (__int64)a7 <= a5 )
    result = a7;
  if ( a4 <= (__int64)result )
  {
LABEL_22:
    if ( v11 != v9 )
      result = memmove(a6, v9, (char *)v11 - (char *)v9);
    v22 = (unsigned __int64 *)((char *)a6 + (char *)v11 - (char *)v9);
    if ( a6 != v22 )
    {
      while ( (unsigned __int64 *)a3 != v11 )
      {
        v61 = sub_DA4700(*a8, *a8[1], *v11, *a6, (__int64)a8[2], 0);
        if ( BYTE4(v61) && (int)v61 < 0 )
        {
          result = (void *)*v11;
          ++v9;
          ++v11;
          *(v9 - 1) = (unsigned __int64)result;
          if ( v22 == a6 )
            return result;
        }
        else
        {
          result = (void *)*a6++;
          *v9++ = (unsigned __int64)result;
          if ( v22 == a6 )
            return result;
        }
      }
    }
    if ( v22 != a6 )
    {
      v28 = a6;
      v29 = v9;
      v23 = (char *)v22 - (char *)a6;
      return memmove(v29, v28, v23);
    }
  }
  else
  {
    if ( (__int64)a7 < a5 )
    {
      v51 = a1;
      v12 = a4;
      v13 = a2;
      v14 = a5;
      dest = a6;
      while ( 1 )
      {
        src = v13;
        if ( v12 > v14 )
        {
          v30 = sub_DA4CA0(v13, a3, &v51[v12 / 2], (__int64)a8);
          v17 = (char *)src;
          v18 = (char *)&v51[v12 / 2];
          v50 = v30;
          v19 = v12 / 2;
          v15 = v30 - src;
        }
        else
        {
          v15 = v14 / 2;
          v50 = &v13[v14 / 2];
          v16 = sub_DA4C00(v51, (__int64)v13, v50, (__int64)a8);
          v17 = (char *)src;
          v18 = (char *)v16;
          v19 = v16 - v51;
        }
        v12 -= v19;
        if ( v12 <= v15 || v15 > (__int64)a7 )
        {
          if ( v12 > (__int64)a7 )
          {
            v41 = v19;
            v48 = (int)v18;
            v34 = sub_D92500(v18, v17, (char *)v50);
            LODWORD(v19) = v41;
            LODWORD(v18) = v48;
            srca = v34;
          }
          else
          {
            srca = (char *)v50;
            if ( v12 )
            {
              v31 = v17 - v18;
              if ( v17 != v18 )
              {
                v36 = v17;
                v39 = v19;
                v45 = v17 - v18;
                srcf = v18;
                memmove(dest, v18, v17 - v18);
                v17 = v36;
                LODWORD(v19) = v39;
                v31 = v45;
                v18 = srcf;
              }
              if ( v17 != (char *)v50 )
              {
                v46 = v31;
                srcg = v19;
                v32 = (unsigned int)memmove(v18, v17, (char *)v50 - v17);
                v31 = v46;
                LODWORD(v19) = srcg;
                LODWORD(v18) = v32;
              }
              srca = (char *)v50 - v31;
              if ( v31 )
              {
                v40 = (int)v18;
                v47 = v19;
                memmove((char *)v50 - v31, dest, v31);
                LODWORD(v19) = v47;
                LODWORD(v18) = v40;
              }
            }
          }
        }
        else
        {
          srca = v18;
          if ( v15 )
          {
            v20 = (char *)v50 - v17;
            if ( v17 != (char *)v50 )
            {
              v35 = v18;
              v37 = v19;
              v42 = (char *)v50 - v17;
              srcc = v17;
              memmove(dest, v17, (char *)v50 - v17);
              v18 = v35;
              LODWORD(v19) = v37;
              v20 = v42;
              v17 = srcc;
            }
            if ( v17 != v18 )
            {
              v38 = v20;
              v43 = v19;
              srcd = v18;
              memmove((char *)v50 - (v17 - v18), v18, v17 - v18);
              v20 = v38;
              LODWORD(v19) = v43;
              v18 = srcd;
            }
            if ( v20 )
            {
              v44 = v19;
              srce = (void *)v20;
              v21 = (char *)memmove(v18, dest, v20);
              LODWORD(v19) = v44;
              v20 = (size_t)srce;
              v18 = v21;
            }
            srca = &v18[v20];
          }
        }
        v14 -= v15;
        sub_DA5350((_DWORD)v51, (_DWORD)v18, (_DWORD)srca, v19, v15, (_DWORD)dest, (__int64)a7, (__int64)a8);
        result = a7;
        if ( v14 <= (__int64)a7 )
          result = (void *)v14;
        if ( v12 <= (__int64)result )
        {
          a6 = dest;
          v11 = v50;
          v9 = (unsigned __int64 *)srca;
          goto LABEL_22;
        }
        if ( v14 <= (__int64)a7 )
          break;
        v13 = v50;
        v51 = (unsigned __int64 *)srca;
      }
      a6 = dest;
      v11 = v50;
      v9 = (unsigned __int64 *)srca;
    }
    result = (void *)a3;
    v23 = a3 - (_QWORD)v11;
    if ( (unsigned __int64 *)a3 != v11 )
    {
      result = memmove(a6, v11, v23);
      v23 = a3 - (_QWORD)v11;
    }
    v24 = (unsigned __int64 *)((char *)a6 + v23);
    if ( v11 == v9 )
    {
      if ( a6 != v24 )
      {
        v33 = (_QWORD *)a3;
        goto LABEL_59;
      }
    }
    else if ( a6 != v24 )
    {
      srcb = v9;
      v25 = v24 - 1;
      v26 = v11 - 1;
      v27 = (_QWORD *)a3;
      while ( 1 )
      {
        while ( 1 )
        {
          --v27;
          v62 = sub_DA4700(*a8, *a8[1], *v25, *v26, (__int64)a8[2], 0);
          if ( BYTE4(v62) )
          {
            if ( (int)v62 < 0 )
              break;
          }
          result = (void *)*v25;
          *v27 = *v25;
          if ( a6 == v25 )
            return result;
          --v25;
        }
        result = (void *)*v26;
        *v27 = *v26;
        if ( v26 == srcb )
          break;
        --v26;
      }
      if ( a6 != v25 + 1 )
      {
        v23 = (char *)(v25 + 1) - (char *)a6;
        v33 = v27;
LABEL_59:
        v29 = (_QWORD *)((char *)v33 - v23);
        v28 = a6;
        return memmove(v29, v28, v23);
      }
    }
  }
  return result;
}
