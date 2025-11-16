// Function: sub_2251C20
// Address: 0x2251c20
//
__int64 __fastcall sub_2251C20(__int64 a1, size_t a2, unsigned __int64 a3, const wchar_t *a4, size_t a5)
{
  unsigned __int64 v6; // rax
  __int64 v7; // rdx
  const wchar_t *v8; // rax
  size_t v10; // r15
  unsigned __int64 v12; // rbx
  unsigned __int64 v13; // rcx
  wchar_t *v14; // r9
  unsigned __int64 v15; // rsi
  size_t v16; // r10
  bool v17; // bp
  const wchar_t *v18; // rsi
  wchar_t *v19; // rdi
  const wchar_t *v20; // rax
  const wchar_t *v22; // rsi
  wchar_t *v23; // rdi
  __int64 v24; // rcx
  unsigned __int64 v25; // rdi
  const wchar_t *v26; // rsi
  size_t v27; // r13
  size_t v28; // r8
  const wchar_t *v29; // rsi
  wchar_t *v30; // rdi
  wchar_t *s1; // [rsp+8h] [rbp-50h]
  wchar_t *s1a; // [rsp+8h] [rbp-50h]
  wchar_t *s1b; // [rsp+8h] [rbp-50h]
  wchar_t *s1c; // [rsp+8h] [rbp-50h]
  size_t v35; // [rsp+10h] [rbp-48h]
  size_t v36; // [rsp+10h] [rbp-48h]
  size_t v37; // [rsp+10h] [rbp-48h]
  size_t v38; // [rsp+10h] [rbp-48h]
  size_t n; // [rsp+18h] [rbp-40h]
  size_t na; // [rsp+18h] [rbp-40h]

  v6 = a3 + 0xFFFFFFFFFFFFFFFLL;
  v7 = *(_QWORD *)(a1 + 8);
  if ( a5 > v6 - v7 )
    sub_4262D8((__int64)"basic_string::_M_replace");
  v8 = *(const wchar_t **)a1;
  v10 = a5 - a3;
  v12 = v7 + a5 - a3;
  if ( *(_QWORD *)a1 == a1 + 16 )
    v13 = 3;
  else
    v13 = *(_QWORD *)(a1 + 16);
  if ( v13 < v12 )
  {
    sub_2251880((const wchar_t **)a1, a2, a3, a4, a5);
    goto LABEL_12;
  }
  v14 = (wchar_t *)&v8[a2];
  v15 = a3 + a2;
  v16 = v7 - v15;
  v17 = a3 != a5 && v7 != v15;
  if ( v8 <= a4 && a4 <= &v8[v7] )
  {
    if ( a5 && a3 >= a5 )
    {
      if ( a5 != 1 )
      {
        n = v7 - v15;
        v36 = a5;
        s1a = v14;
        wmemmove(v14, a4, a5);
        v14 = s1a;
        a5 = v36;
        v16 = n;
        if ( !v17 )
          goto LABEL_12;
LABEL_18:
        v22 = &v14[a3];
        v23 = &v14[a5];
        if ( v16 == 1 )
        {
          *v23 = *v22;
        }
        else
        {
          v37 = a5;
          s1b = v14;
          wmemmove(v23, v22, v16);
          a5 = v37;
          v14 = s1b;
        }
LABEL_20:
        if ( a3 >= a5 )
          goto LABEL_12;
        v24 = 4 * a5;
        v25 = (unsigned __int64)&v14[a3];
        if ( v25 < (unsigned __int64)&a4[a5] )
        {
          if ( v25 > (unsigned __int64)a4 )
          {
            v27 = (__int64)(v25 - (_QWORD)a4) >> 2;
            if ( v27 == 1 )
            {
              *v14 = *a4;
            }
            else if ( v27 )
            {
              na = a5;
              v38 = 4 * a5;
              s1c = v14;
              wmemmove(v14, a4, v27);
              a5 = na;
              v24 = v38;
              v14 = s1c;
            }
            v28 = a5 - v27;
            v29 = (wchar_t *)((char *)v14 + v24);
            v30 = (wchar_t *)((char *)v14 + v25 - (_QWORD)a4);
            if ( v28 == 1 )
            {
              *v30 = *v29;
            }
            else if ( v28 )
            {
              wmemcpy(v30, v29, v28);
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
              wmemcpy(v14, v26, a5);
            }
          }
          goto LABEL_12;
        }
        if ( a5 != 1 )
        {
          if ( a5 )
            wmemmove(v14, a4, a5);
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
      v35 = a5;
      s1 = v14;
      wmemmove(v19, v18, v16);
      a5 = v35;
      v14 = s1;
    }
  }
  if ( a5 )
  {
    if ( a5 != 1 )
    {
      wmemcpy(v14, a4, a5);
      goto LABEL_12;
    }
    goto LABEL_27;
  }
LABEL_12:
  v20 = *(const wchar_t **)a1;
  *(_QWORD *)(a1 + 8) = v12;
  v20[v12] = 0;
  return a1;
}
