// Function: sub_1834080
// Address: 0x1834080
//
_QWORD *__fastcall sub_1834080(__int64 a1, unsigned __int64 *a2)
{
  _QWORD *v4; // rbx
  unsigned __int64 v5; // rdi
  unsigned __int64 v6; // rsi
  char *v7; // rcx
  char *v8; // rax
  _QWORD *v9; // rdx
  signed __int64 v10; // r8
  _QWORD *v11; // rax
  char v12; // dl
  char *v13; // rsi
  char *v14; // rax
  _QWORD *v15; // rdx
  signed __int64 v16; // rdi
  _QWORD *result; // rax
  __int64 v18; // rax

  v4 = *(_QWORD **)(a1 + 16);
  if ( !v4 )
  {
    v4 = (_QWORD *)(a1 + 8);
    goto LABEL_29;
  }
  v5 = *a2;
  while ( 1 )
  {
    v6 = v4[4];
    if ( v5 < v6 )
      goto LABEL_12;
    if ( v5 != v6 )
      goto LABEL_15;
    v7 = (char *)a2[2];
    v8 = (char *)a2[1];
    v9 = (_QWORD *)v4[5];
    v10 = v4[6] - (_QWORD)v9;
    if ( v7 - v8 > v10 )
      v7 = &v8[v10];
    if ( v8 == v7 )
      break;
    while ( *(_QWORD *)v8 >= *v9 )
    {
      if ( *(_QWORD *)v8 > *v9 )
        goto LABEL_15;
      v8 += 8;
      ++v9;
      if ( v7 == v8 )
        goto LABEL_14;
    }
LABEL_12:
    v11 = (_QWORD *)v4[2];
    v12 = 1;
    if ( !v11 )
      goto LABEL_16;
LABEL_13:
    v4 = v11;
  }
LABEL_14:
  if ( (_QWORD *)v4[6] != v9 )
    goto LABEL_12;
LABEL_15:
  v11 = (_QWORD *)v4[3];
  v12 = 0;
  if ( v11 )
    goto LABEL_13;
LABEL_16:
  if ( !v12 )
    goto LABEL_17;
LABEL_29:
  result = 0;
  if ( *(_QWORD **)(a1 + 24) != v4 )
  {
    v18 = sub_220EF80(v4);
    v5 = *a2;
    v6 = *(_QWORD *)(v18 + 32);
    v4 = (_QWORD *)v18;
LABEL_17:
    if ( v5 <= v6 )
    {
      if ( v5 != v6 )
        return v4;
      v13 = (char *)v4[6];
      v14 = (char *)v4[5];
      v15 = (_QWORD *)a2[1];
      v16 = a2[2] - (_QWORD)v15;
      if ( v13 - v14 > v16 )
        v13 = &v14[v16];
      if ( v14 == v13 )
      {
LABEL_31:
        if ( (_QWORD *)a2[2] == v15 )
          return v4;
      }
      else
      {
        while ( *(_QWORD *)v14 >= *v15 )
        {
          if ( *(_QWORD *)v14 > *v15 )
            return v4;
          v14 += 8;
          ++v15;
          if ( v13 == v14 )
            goto LABEL_31;
        }
      }
    }
    return 0;
  }
  return result;
}
