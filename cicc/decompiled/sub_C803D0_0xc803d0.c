// Function: sub_C803D0
// Address: 0xc803d0
//
__int64 *__fastcall sub_C803D0(__int64 *a1)
{
  unsigned __int64 v2; // rdx
  unsigned __int64 v3; // rax
  __int64 v4; // r14
  const char *v5; // r13
  size_t v6; // rax
  unsigned __int64 v7; // rax
  unsigned __int64 v8; // rcx
  __int64 v9; // rdx
  unsigned __int64 v10; // rsi
  unsigned __int64 v11; // rdi
  unsigned int v13; // esi
  __int64 v14; // rax
  _BYTE *v15; // rax
  __int64 v16; // rax
  unsigned __int64 v17; // rsi
  unsigned __int64 v18; // rdx
  _BOOL8 v19; // rcx
  __int64 v20; // rax
  bool v21; // zf
  __int64 v22; // rdx

  v2 = a1[3];
  v3 = v2 + a1[4];
  a1[4] = v3;
  if ( v3 == a1[1] )
  {
    a1[2] = 0;
    a1[3] = 0;
    return a1;
  }
  if ( v2 <= 2 )
    goto LABEL_3;
  if ( !sub_C80220(*(_BYTE *)a1[2], *((_DWORD *)a1 + 10))
    || (v15 = (_BYTE *)a1[2], v15[1] != *v15)
    || sub_C80220(v15[2], *((_DWORD *)a1 + 10)) )
  {
    v3 = a1[4];
LABEL_3:
    if ( !sub_C80220(*(_BYTE *)(*a1 + v3), *((_DWORD *)a1 + 10)) )
      goto LABEL_4;
    v13 = *((_DWORD *)a1 + 10);
    if ( v13 <= 1 || (v16 = a1[3]) == 0 || *(_BYTE *)(a1[2] + v16 - 1) != 58 )
    {
      v14 = a1[4];
      if ( a1[1] == v14 )
      {
        v22 = a1[4];
      }
      else
      {
        while ( sub_C80220(*(_BYTE *)(*a1 + v14), v13) )
        {
          v22 = a1[1];
          v14 = a1[4] + 1;
          a1[4] = v14;
          if ( v14 == v22 )
            goto LABEL_22;
          v13 = *((_DWORD *)a1 + 10);
        }
        v4 = a1[4];
        v22 = a1[1];
        if ( v4 != v22 )
          goto LABEL_5;
      }
LABEL_22:
      if ( a1[3] != 1 || *(_BYTE *)a1[2] != 47 )
      {
        a1[3] = 1;
        a1[4] = v22 - 1;
        a1[2] = (__int64)".";
        return a1;
      }
LABEL_4:
      v4 = a1[4];
LABEL_5:
      v5 = "/";
      if ( *((_DWORD *)a1 + 10) >= 2u )
        v5 = "\\/";
      v6 = strlen(v5);
      v7 = sub_C934D0(a1, v5, v6, v4);
      v8 = a1[1];
      v9 = *a1;
      v10 = v8;
      if ( a1[4] <= v8 )
        v10 = a1[4];
      v11 = 0;
      if ( v7 >= v10 )
      {
        if ( v7 <= v8 )
          v8 = v7;
        v11 = v8 - v10;
      }
      a1[3] = v11;
      a1[2] = v10 + v9;
      return a1;
    }
    goto LABEL_30;
  }
  if ( !sub_C80220(*(_BYTE *)(*a1 + a1[4]), *((_DWORD *)a1 + 10)) )
    goto LABEL_4;
LABEL_30:
  v17 = a1[4];
  v18 = a1[1];
  v19 = 0;
  v20 = *a1;
  v21 = v17 == v18;
  if ( v17 <= v18 )
  {
    v18 = a1[4];
    v19 = !v21;
  }
  a1[3] = v19;
  a1[2] = v18 + v20;
  return a1;
}
