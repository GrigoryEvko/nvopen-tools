// Function: sub_2241130
// Address: 0x2241130
//
unsigned __int64 *__fastcall sub_2241130(unsigned __int64 *a1, size_t a2, unsigned __int64 a3, _BYTE *a4, size_t a5)
{
  unsigned __int64 v6; // rax
  unsigned __int64 v7; // rdx
  unsigned __int64 v8; // rax
  size_t v10; // r15
  unsigned __int64 v12; // rbx
  unsigned __int64 v13; // rcx
  _BYTE *v14; // r9
  unsigned __int64 v15; // rsi
  size_t v16; // r10
  bool v17; // bp
  _BYTE *v18; // rsi
  _BYTE *v19; // rdi
  unsigned __int64 v20; // rax
  _BYTE *v22; // rax
  _BYTE *v23; // rsi
  _BYTE *v24; // rdi
  _BYTE *v25; // r13
  _BYTE *v26; // rsi
  size_t v27; // r13
  _BYTE *v28; // rax
  _BYTE *v29; // rsi
  _BYTE *v30; // rdi
  size_t v31; // rdx
  _BYTE *dest; // [rsp+0h] [rbp-48h]
  void *desta; // [rsp+0h] [rbp-48h]
  _BYTE *destb; // [rsp+0h] [rbp-48h]
  void *destc; // [rsp+0h] [rbp-48h]
  size_t n; // [rsp+8h] [rbp-40h]
  size_t na; // [rsp+8h] [rbp-40h]
  size_t nb; // [rsp+8h] [rbp-40h]

  v6 = a3 + 0x3FFFFFFFFFFFFFFFLL;
  v7 = a1[1];
  if ( a5 > v6 - v7 )
    sub_4262D8((__int64)"basic_string::_M_replace");
  v8 = *a1;
  v10 = a5 - a3;
  v12 = v7 + a5 - a3;
  if ( (unsigned __int64 *)*a1 == a1 + 2 )
    v13 = 15;
  else
    v13 = a1[2];
  if ( v13 < v12 )
  {
    sub_2240BB0(a1, a2, a3, a4, a5);
    goto LABEL_12;
  }
  v14 = (_BYTE *)(v8 + a2);
  v15 = a3 + a2;
  v16 = v7 - v15;
  v17 = a3 != a5 && v7 != v15;
  if ( v8 <= (unsigned __int64)a4 && (unsigned __int64)a4 <= v7 + v8 )
  {
    if ( a5 && a3 >= a5 )
    {
      if ( a5 != 1 )
      {
        na = v7 - v15;
        desta = (void *)a5;
        v22 = memmove(v14, a4, a5);
        a5 = (size_t)desta;
        v16 = na;
        v14 = v22;
        if ( !v17 )
          goto LABEL_12;
LABEL_18:
        v23 = &v14[a3];
        v24 = &v14[a5];
        if ( v16 == 1 )
        {
          *v24 = *v23;
        }
        else
        {
          nb = a5;
          destb = v14;
          memmove(v24, v23, v16);
          a5 = nb;
          v14 = destb;
        }
LABEL_20:
        if ( a3 >= a5 )
          goto LABEL_12;
        v25 = &v14[a3];
        if ( v25 < &a4[a5] )
        {
          if ( v25 > a4 )
          {
            v27 = v25 - a4;
            if ( v27 == 1 )
            {
              *v14 = *a4;
            }
            else if ( v27 )
            {
              destc = (void *)a5;
              v28 = memmove(v14, a4, v27);
              a5 = (size_t)destc;
              v14 = v28;
            }
            v29 = &v14[a5];
            v30 = &v14[v27];
            v31 = a5 - v27;
            if ( a5 - v27 == 1 )
            {
              *v30 = *v29;
            }
            else if ( v31 )
            {
              memcpy(v30, v29, v31);
            }
          }
          else
          {
            v26 = &a4[v10];
            if ( a5 == 1 )
            {
              *v14 = *v26;
            }
            else if ( a5 )
            {
              memcpy(v14, v26, a5);
            }
          }
          goto LABEL_12;
        }
        if ( a5 != 1 )
        {
          if ( a5 )
            memmove(v14, a4, a5);
          goto LABEL_12;
        }
LABEL_27:
        *v14 = *a4;
        goto LABEL_12;
      }
      *v14 = *a4;
    }
    if ( !v17 )
      goto LABEL_20;
    goto LABEL_18;
  }
  if ( v17 )
  {
    v18 = &v14[a3];
    v19 = &v14[a5];
    if ( v16 == 1 )
    {
      *v19 = *v18;
    }
    else
    {
      n = a5;
      dest = v14;
      memmove(v19, v18, v16);
      a5 = n;
      v14 = dest;
    }
  }
  if ( a5 )
  {
    if ( a5 != 1 )
    {
      memcpy(v14, a4, a5);
      goto LABEL_12;
    }
    goto LABEL_27;
  }
LABEL_12:
  v20 = *a1;
  a1[1] = v12;
  *(_BYTE *)(v20 + v12) = 0;
  return a1;
}
