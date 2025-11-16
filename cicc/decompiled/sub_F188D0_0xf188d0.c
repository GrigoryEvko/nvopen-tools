// Function: sub_F188D0
// Address: 0xf188d0
//
void *__fastcall sub_F188D0(
        char *a1,
        char *a2,
        __int64 a3,
        __int64 a4,
        __int64 a5,
        void **a6,
        void *a7,
        unsigned __int8 (__fastcall *a8)(_QWORD, void *))
{
  char *v9; // r13
  char *v10; // r12
  char *v11; // rbx
  void *result; // rax
  __int64 v13; // r15
  char *v14; // r9
  __int64 v15; // rbx
  __int64 v16; // r14
  char *v17; // rax
  char *v18; // r9
  char *v19; // r12
  __int64 v20; // r11
  char *v21; // r10
  size_t v22; // r10
  void **v23; // r15
  char *v24; // rax
  char *v25; // rbx
  char *v26; // r13
  size_t v27; // rdx
  void **v28; // r15
  char *v29; // r12
  void **v30; // rsi
  char *v31; // rdi
  char *v32; // rax
  size_t v33; // r8
  char *v34; // rax
  char *v35; // rax
  int v36; // [rsp+0h] [rbp-70h]
  char *v37; // [rsp+0h] [rbp-70h]
  size_t v38; // [rsp+8h] [rbp-68h]
  size_t v39; // [rsp+8h] [rbp-68h]
  int v40; // [rsp+8h] [rbp-68h]
  int v41; // [rsp+8h] [rbp-68h]
  size_t v42; // [rsp+8h] [rbp-68h]
  char *src; // [rsp+18h] [rbp-58h]
  char *srcb; // [rsp+18h] [rbp-58h]
  int srcc; // [rsp+18h] [rbp-58h]
  void *srcd; // [rsp+18h] [rbp-58h]
  char *srca; // [rsp+18h] [rbp-58h]
  void *srce; // [rsp+18h] [rbp-58h]
  int srcf; // [rsp+18h] [rbp-58h]
  int srcg; // [rsp+18h] [rbp-58h]
  int srch; // [rsp+18h] [rbp-58h]
  void **dest; // [rsp+20h] [rbp-50h]
  char *v54; // [rsp+28h] [rbp-48h]
  char *v55; // [rsp+30h] [rbp-40h]

  v9 = a1;
  v10 = a2;
  v11 = (char *)a3;
  result = (void *)a5;
  if ( (__int64)a7 <= a5 )
    result = a7;
  if ( a4 <= (__int64)result )
  {
LABEL_22:
    if ( v10 != v9 )
      result = memmove(a6, v9, v10 - v9);
    v23 = (void **)((char *)a6 + v10 - v9);
    if ( a6 != v23 && v11 != v10 )
    {
      v24 = v11;
      v25 = v9;
      v26 = v24;
      do
      {
        if ( a8(*(_QWORD *)v10, *a6) )
        {
          result = *(void **)v10;
          v25 += 8;
          v10 += 8;
          *((_QWORD *)v25 - 1) = result;
          if ( v23 == a6 )
            return result;
        }
        else
        {
          result = *a6++;
          v25 += 8;
          *((_QWORD *)v25 - 1) = result;
          if ( v23 == a6 )
            return result;
        }
      }
      while ( v26 != v10 );
      v9 = v25;
    }
    if ( v23 != a6 )
    {
      v30 = a6;
      v31 = v9;
      v27 = (char *)v23 - (char *)a6;
      return memmove(v31, v30, v27);
    }
  }
  else
  {
    v13 = a5;
    if ( (__int64)a7 < a5 )
    {
      v55 = a1;
      v14 = a2;
      v15 = a4;
      dest = a6;
      while ( 1 )
      {
        src = v14;
        if ( v15 > v13 )
        {
          v19 = &v55[8 * (v15 / 2)];
          v32 = (char *)sub_F185B0(v14, a3, v19, (unsigned __int8 (__fastcall *)(_QWORD, _QWORD))a8);
          v18 = src;
          v20 = v15 / 2;
          v54 = v32;
          v16 = (v32 - src) >> 3;
        }
        else
        {
          v16 = v13 / 2;
          v54 = &v14[8 * (v13 / 2)];
          v17 = (char *)sub_F18630(v55, (__int64)v14, v54, (unsigned __int8 (__fastcall *)(_QWORD, _QWORD))a8);
          v18 = src;
          v19 = v17;
          v20 = (v17 - v55) >> 3;
        }
        v15 -= v20;
        if ( v15 <= v16 || v16 > (__int64)a7 )
        {
          if ( v15 > (__int64)a7 )
          {
            srch = v20;
            v35 = sub_F07C40(v19, v18, v54);
            LODWORD(v20) = srch;
            v21 = v35;
          }
          else
          {
            v21 = v54;
            if ( v15 )
            {
              v33 = v18 - v19;
              if ( v18 != v19 )
              {
                v37 = v18;
                v41 = v20;
                srce = (void *)(v18 - v19);
                memmove(dest, v19, v18 - v19);
                v18 = v37;
                LODWORD(v20) = v41;
                v33 = (size_t)srce;
              }
              if ( v18 != v54 )
              {
                v42 = v33;
                srcf = v20;
                memmove(v19, v18, v54 - v18);
                v33 = v42;
                LODWORD(v20) = srcf;
              }
              v21 = &v54[-v33];
              if ( v33 )
              {
                srcg = v20;
                v34 = (char *)memmove(&v54[-v33], dest, v33);
                LODWORD(v20) = srcg;
                v21 = v34;
              }
            }
          }
        }
        else
        {
          v21 = v19;
          if ( v16 )
          {
            v22 = v54 - v18;
            if ( v18 != v54 )
            {
              v36 = v20;
              v38 = v54 - v18;
              srcb = v18;
              memmove(dest, v18, v54 - v18);
              LODWORD(v20) = v36;
              v22 = v38;
              v18 = srcb;
            }
            if ( v18 != v19 )
            {
              v39 = v22;
              srcc = v20;
              memmove(&v54[-(v18 - v19)], v19, v18 - v19);
              v22 = v39;
              LODWORD(v20) = srcc;
            }
            if ( v22 )
            {
              v40 = v20;
              srcd = (void *)v22;
              memmove(v19, dest, v22);
              LODWORD(v20) = v40;
              v22 = (size_t)srcd;
            }
            v21 = &v19[v22];
          }
        }
        v13 -= v16;
        srca = v21;
        sub_F188D0((_DWORD)v55, (_DWORD)v19, (_DWORD)v21, v20, v16, (_DWORD)dest, (__int64)a7, (__int64)a8);
        result = a7;
        if ( v13 <= (__int64)a7 )
          result = (void *)v13;
        if ( v15 <= (__int64)result )
        {
          v11 = (char *)a3;
          a6 = dest;
          v9 = srca;
          v10 = v54;
          goto LABEL_22;
        }
        if ( v13 <= (__int64)a7 )
          break;
        v55 = srca;
        v14 = v54;
      }
      v11 = (char *)a3;
      a6 = dest;
      v9 = srca;
      v10 = v54;
    }
    v27 = v11 - v10;
    if ( v11 != v10 )
    {
      result = memmove(a6, v10, v27);
      v27 = v11 - v10;
    }
    v28 = (void **)((char *)a6 + v27);
    if ( v10 == v9 )
    {
      if ( a6 != v28 )
      {
LABEL_60:
        v30 = a6;
        v31 = &v11[-v27];
        return memmove(v31, v30, v27);
      }
    }
    else if ( a6 != v28 )
    {
      v29 = v10 - 8;
LABEL_39:
      --v28;
      while ( 1 )
      {
        v11 -= 8;
        if ( !a8(*v28, *(void **)v29) )
        {
          result = *v28;
          *(_QWORD *)v11 = *v28;
          if ( a6 != v28 )
            goto LABEL_39;
          return result;
        }
        result = *(void **)v29;
        *(_QWORD *)v11 = *(_QWORD *)v29;
        if ( v29 == v9 )
          break;
        v29 -= 8;
      }
      if ( a6 == v28 + 1 )
        return result;
      v27 = (char *)(v28 + 1) - (char *)a6;
      goto LABEL_60;
    }
  }
  return result;
}
