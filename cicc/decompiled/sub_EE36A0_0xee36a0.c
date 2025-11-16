// Function: sub_EE36A0
// Address: 0xee36a0
//
__int64 __fastcall sub_EE36A0(_QWORD *a1, const void *a2)
{
  _QWORD *v2; // r13
  _QWORD *v4; // r12
  _QWORD *v5; // rax
  char *v6; // r14
  char *v7; // rdi
  __int64 v8; // rax
  __int64 v9; // rdx
  __int64 v10; // rax
  _QWORD *v11; // r12
  _QWORD *v12; // rax
  char *v13; // rdi
  char *v14; // r15
  __int64 v15; // rax
  __int64 result; // rax
  _QWORD *v17; // rdi
  _QWORD *v18; // rdi
  size_t v19; // rdx
  size_t v20; // rdx

  v2 = a1 + 4;
  v4 = (_QWORD *)*a1;
  v5 = (_QWORD *)a1[1];
  v6 = (char *)(*a1 + 688LL);
  v7 = *(char **)(*a1 + 664LL);
  if ( v5 == v2 )
  {
    a2 = v2;
    if ( v7 != v6 )
    {
      _libc_free(v7, v2);
      v4[83] = v6;
      v4[84] = v6;
      v4[85] = v4 + 90;
      a2 = (const void *)a1[1];
    }
    v20 = a1[2] - (_QWORD)a2;
    if ( (const void *)a1[2] != a2 )
    {
      memmove(v6, a2, v20);
      v6 = (char *)v4[83];
      v20 = a1[2] - a1[1];
    }
    v4[84] = &v6[v20];
    a1[2] = a1[1];
  }
  else
  {
    v4[83] = v5;
    if ( v7 == v6 )
    {
      v4[84] = a1[2];
      v4[85] = a1[3];
      a1[1] = v2;
      a1[2] = v2;
      a1[3] = a1 + 8;
    }
    else
    {
      a1[1] = v7;
      v8 = v4[84];
      v4[84] = a1[2];
      v9 = a1[3];
      a1[2] = v8;
      v10 = v4[85];
      v4[85] = v9;
      a1[3] = v10;
      a1[2] = a1[1];
    }
  }
  v11 = (_QWORD *)*a1;
  v12 = (_QWORD *)a1[8];
  v13 = *(char **)(*a1 + 576LL);
  v14 = (char *)(*a1 + 600LL);
  if ( v12 == a1 + 11 )
  {
    a2 = a1 + 11;
    if ( v14 != v13 )
    {
      _libc_free(v13, a2);
      v11[72] = v14;
      v11[73] = v14;
      v11[74] = v11 + 83;
      a2 = (const void *)a1[8];
    }
    result = a1[9];
    v19 = result - (_QWORD)a2;
    if ( a2 != (const void *)result )
    {
      result = (__int64)memmove(v14, a2, v19);
      v14 = (char *)v11[72];
      v19 = a1[9] - a1[8];
    }
    v11[73] = &v14[v19];
  }
  else
  {
    v11[72] = v12;
    if ( v14 == v13 )
    {
      v11[73] = a1[9];
      result = a1[10];
      v11[74] = result;
      goto LABEL_9;
    }
    v15 = a1[9];
    a1[8] = v13;
    v11[73] = v15;
    result = a1[10];
    v11[74] = result;
  }
  v17 = (_QWORD *)a1[8];
  if ( v17 != a1 + 11 )
    result = _libc_free(v17, a2);
LABEL_9:
  v18 = (_QWORD *)a1[1];
  if ( v18 != v2 )
    return _libc_free(v18, a2);
  return result;
}
