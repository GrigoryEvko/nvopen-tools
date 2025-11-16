// Function: sub_1977D40
// Address: 0x1977d40
//
void *__fastcall sub_1977D40(__int64 a1, __int64 a2)
{
  _QWORD *v3; // rsi
  char *v4; // r8
  _QWORD *i; // rdx
  _QWORD *v6; // rax
  __int64 v7; // rdx
  __int64 v8; // rcx
  _QWORD *v9; // rdx
  _QWORD *v10; // rbx
  void *result; // rax

  v3 = *(_QWORD **)(a1 + 8);
  v4 = *(char **)(a1 + 16);
  for ( i = v3; ; ++i )
  {
    v6 = (_QWORD *)*i;
    if ( *i == a2 )
      break;
  }
  v7 = (v4 - (char *)v3) >> 5;
  v8 = (v4 - (char *)v3) >> 3;
  if ( v7 <= 0 )
  {
LABEL_13:
    if ( v8 != 2 )
    {
      if ( v8 != 3 )
      {
        if ( v8 != 1 )
        {
LABEL_16:
          v10 = *(_QWORD **)v4;
          v3 = *(_QWORD **)(a1 + 16);
          goto LABEL_11;
        }
LABEL_21:
        v10 = (_QWORD *)*v3;
        if ( v6 == (_QWORD *)*v3 )
          goto LABEL_11;
        goto LABEL_16;
      }
      v10 = (_QWORD *)*v3;
      if ( v6 == (_QWORD *)*v3 )
        goto LABEL_11;
      ++v3;
    }
    v10 = (_QWORD *)*v3;
    if ( v6 == (_QWORD *)*v3 )
      goto LABEL_11;
    ++v3;
    goto LABEL_21;
  }
  v9 = &v3[4 * v7];
  while ( 1 )
  {
    v10 = (_QWORD *)*v3;
    if ( v6 == (_QWORD *)*v3 )
      break;
    v10 = (_QWORD *)v3[1];
    if ( v6 == v10 )
    {
      ++v3;
      break;
    }
    v10 = (_QWORD *)v3[2];
    if ( v6 == v10 )
    {
      v3 += 2;
      break;
    }
    v10 = (_QWORD *)v3[3];
    if ( v6 == v10 )
    {
      v3 += 3;
      break;
    }
    v3 += 4;
    if ( v3 == v9 )
    {
      v8 = (v4 - (char *)v3) >> 3;
      goto LABEL_13;
    }
  }
LABEL_11:
  result = sub_13FDAF0(a1 + 8, v3);
  *v10 = 0;
  return result;
}
