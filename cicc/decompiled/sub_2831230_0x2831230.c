// Function: sub_2831230
// Address: 0x2831230
//
void *__fastcall sub_2831230(__int64 a1, __int64 a2)
{
  char *v4; // rsi
  char *v5; // rdi
  char *v6; // rdx
  _QWORD *v7; // rax
  __int64 v8; // rdx
  __int64 v9; // rcx
  char *v10; // rdx
  _QWORD *v11; // rbx
  void *result; // rax

  v4 = *(char **)(a1 + 8);
  v5 = *(char **)(a1 + 16);
  if ( v4 == v5 )
LABEL_27:
    BUG();
  v6 = v4;
  while ( 1 )
  {
    v7 = *(_QWORD **)v6;
    if ( *(_QWORD *)v6 == a2 )
      break;
    v6 += 8;
    if ( v5 == v6 )
      goto LABEL_27;
  }
  v8 = (v5 - v4) >> 5;
  v9 = (v5 - v4) >> 3;
  if ( v8 <= 0 )
  {
LABEL_14:
    if ( v9 != 2 )
    {
      if ( v9 != 3 )
      {
        if ( v9 != 1 )
        {
LABEL_17:
          v11 = *(_QWORD **)v5;
          v4 = v5;
          goto LABEL_12;
        }
LABEL_22:
        v11 = *(_QWORD **)v4;
        if ( v7 == *(_QWORD **)v4 )
          goto LABEL_12;
        goto LABEL_17;
      }
      v11 = *(_QWORD **)v4;
      if ( v7 == *(_QWORD **)v4 )
        goto LABEL_12;
      v4 += 8;
    }
    v11 = *(_QWORD **)v4;
    if ( v7 == *(_QWORD **)v4 )
      goto LABEL_12;
    v4 += 8;
    goto LABEL_22;
  }
  v10 = &v4[32 * v8];
  while ( 1 )
  {
    v11 = *(_QWORD **)v4;
    if ( v7 == *(_QWORD **)v4 )
      break;
    v11 = (_QWORD *)*((_QWORD *)v4 + 1);
    if ( v7 == v11 )
    {
      v4 += 8;
      break;
    }
    v11 = (_QWORD *)*((_QWORD *)v4 + 2);
    if ( v7 == v11 )
    {
      v4 += 16;
      break;
    }
    v11 = (_QWORD *)*((_QWORD *)v4 + 3);
    if ( v7 == v11 )
    {
      v4 += 24;
      break;
    }
    v4 += 32;
    if ( v4 == v10 )
    {
      v9 = (v5 - v4) >> 3;
      goto LABEL_14;
    }
  }
LABEL_12:
  result = sub_D4C9B0(a1 + 8, v4);
  *v11 = 0;
  return result;
}
