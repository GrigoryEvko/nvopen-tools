// Function: sub_EE38E0
// Address: 0xee38e0
//
__int64 __fastcall sub_EE38E0(__int64 *a1, _QWORD *a2)
{
  __int64 *v2; // r15
  char *v3; // r13
  _QWORD *v4; // r12
  const void *v5; // r14
  _BYTE *v7; // rax
  _BYTE *v8; // rsi
  __int64 v9; // rdx
  __int64 v10; // rdx
  const void *v11; // rax
  __int64 v12; // rcx
  __int64 v13; // rdx
  __int64 result; // rax
  size_t v15; // rdx
  _BYTE *v16; // rdx
  char *v17; // rax
  char *v18; // rdi
  char *v19; // rcx
  __int64 v20; // rax
  char *v21; // rcx
  __int64 v22; // rax

  v2 = a1 + 19;
  v3 = (char *)(a1 + 11);
  v4 = a2;
  v5 = a2 + 75;
  a1[3] = (__int64)(a1 + 8);
  v7 = a2 + 86;
  *a1 = (__int64)a2;
  a1[1] = (__int64)(a1 + 4);
  a1[2] = (__int64)(a1 + 4);
  a1[8] = (__int64)(a1 + 11);
  a1[9] = (__int64)(a1 + 11);
  a1[10] = (__int64)(a1 + 19);
  *((_OWORD *)a1 + 2) = 0;
  *((_OWORD *)a1 + 3) = 0;
  *(_OWORD *)(a1 + 11) = 0;
  *(_OWORD *)(a1 + 13) = 0;
  *(_OWORD *)(a1 + 15) = 0;
  *(_OWORD *)(a1 + 17) = 0;
  v8 = (_BYTE *)a2[83];
  if ( v8 == v7 )
  {
    v16 = (_BYTE *)v4[84];
    if ( v8 == v16 )
    {
      v11 = (const void *)v4[72];
      if ( v11 == v5 )
        goto LABEL_8;
      v18 = (char *)(a1 + 11);
    }
    else
    {
      v17 = (char *)memmove(a1 + 4, v8, v16 - v8);
      v18 = (char *)a1[8];
      v19 = v17;
      v20 = v4[83];
      v21 = &v19[v4[84] - v20];
      v4[84] = v20;
      v11 = (const void *)v4[72];
      a1[2] = (__int64)v21;
      if ( v11 == v5 )
      {
        if ( v3 != v18 )
        {
          _libc_free(v18, v8);
          a1[8] = (__int64)v3;
          v5 = (const void *)v4[72];
          a1[9] = (__int64)v3;
          a1[10] = (__int64)v2;
        }
        goto LABEL_8;
      }
    }
    v12 = v4[73];
    v13 = v4[74];
    if ( v3 != v18 )
    {
      a1[8] = (__int64)v11;
      v22 = a1[10];
      v4[72] = v18;
      a1[9] = v12;
      a1[10] = v13;
      v4[74] = v22;
      v4[73] = v18;
      goto LABEL_5;
    }
    goto LABEL_4;
  }
  v9 = v4[84];
  v4[83] = v7;
  v4[84] = v7;
  a1[2] = v9;
  v10 = v4[85];
  v4[85] = v4 + 90;
  v11 = (const void *)v4[72];
  a1[1] = (__int64)v8;
  a1[3] = v10;
  if ( v11 != v5 )
  {
    v12 = v4[73];
    v13 = v4[74];
LABEL_4:
    a1[8] = (__int64)v11;
    a1[9] = v12;
    a1[10] = v13;
    v4[72] = v5;
    v4[73] = v5;
    v4[74] = v4 + 83;
    goto LABEL_5;
  }
LABEL_8:
  v15 = v4[73] - (_QWORD)v5;
  if ( (const void *)v4[73] != v5 )
  {
    memmove(v3, v5, v15);
    v5 = (const void *)v4[72];
    v3 = (char *)a1[8];
    v15 = v4[73] - (_QWORD)v5;
  }
  v4[73] = v5;
  v4 = (_QWORD *)*a1;
  a1[9] = (__int64)&v3[v15];
LABEL_5:
  v4[84] = v4[83];
  result = *a1;
  *(_QWORD *)(*a1 + 584) = *(_QWORD *)(*a1 + 576);
  return result;
}
