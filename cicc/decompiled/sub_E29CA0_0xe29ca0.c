// Function: sub_E29CA0
// Address: 0xe29ca0
//
void *__fastcall sub_E29CA0(__int64 a1, __int64 *a2)
{
  unsigned __int64 v4; // rcx
  _BYTE *v5; // r12
  void *result; // rax
  _BYTE *v7; // r8
  size_t v8; // r13
  char *v9; // rsi
  unsigned __int64 v10; // rax
  __int64 v11; // rdi
  char *v12; // rdx
  unsigned __int64 v13; // rsi
  unsigned __int64 v14; // rax
  __int64 v15; // rax
  char *v16; // rdx
  unsigned __int64 v17; // rax
  char *v18; // rdi
  unsigned __int64 v19; // rsi
  unsigned __int64 v20; // rax
  __int64 v21; // rax
  _BYTE v22[43]; // [rsp+15h] [rbp-2Bh] BYREF

  if ( *(_BYTE *)(a1 + 24) )
  {
    v9 = (char *)a2[1];
    v10 = a2[2];
    v11 = *a2;
    v12 = v9 + 1;
    if ( (unsigned __int64)(v9 + 1) > v10 )
    {
      v13 = (unsigned __int64)(v9 + 993);
      v14 = 2 * v10;
      if ( v13 > v14 )
        a2[2] = v13;
      else
        a2[2] = v14;
      v15 = realloc((void *)v11);
      *a2 = v15;
      v11 = v15;
      if ( !v15 )
        goto LABEL_20;
      v9 = (char *)a2[1];
      v12 = v9 + 1;
    }
    a2[1] = (__int64)v12;
    v9[v11] = 45;
  }
  v4 = *(_QWORD *)(a1 + 16);
  v5 = v22;
  do
  {
    *--v5 = v4 % 0xA + 48;
    result = (void *)v4;
    v4 /= 0xAu;
  }
  while ( (unsigned __int64)result > 9 );
  v7 = (_BYTE *)(v22 - v5);
  v8 = v22 - v5;
  if ( v22 != v5 )
  {
    v16 = (char *)a2[1];
    v17 = a2[2];
    v18 = (char *)*a2;
    if ( &v16[(_QWORD)v7] <= (char *)v17 )
    {
LABEL_17:
      result = memcpy(&v18[(_QWORD)v16], v5, v8);
      a2[1] += v8;
      return result;
    }
    v19 = (unsigned __int64)&v16[(_QWORD)v7 + 992];
    v20 = 2 * v17;
    if ( v19 > v20 )
      a2[2] = v19;
    else
      a2[2] = v20;
    v21 = realloc(v18);
    *a2 = v21;
    v18 = (char *)v21;
    if ( v21 )
    {
      v16 = (char *)a2[1];
      goto LABEL_17;
    }
LABEL_20:
    abort();
  }
  return result;
}
