// Function: sub_17BD140
// Address: 0x17bd140
//
char *__fastcall sub_17BD140(_QWORD *a1, char **a2)
{
  _QWORD *v2; // r14
  _QWORD *v5; // rax
  char *v6; // rsi
  _QWORD *v7; // r12
  __int64 v8; // rcx
  __int64 v9; // rdx
  char *result; // rax
  __int64 v11; // rax
  char *v12; // rcx
  _QWORD *v13; // rsi
  _QWORD *v14; // rax
  _QWORD *v15; // rdx
  _BOOL8 v16; // rdi
  _BYTE *v17; // rsi
  _QWORD *v18; // rdi
  char *v19; // [rsp+8h] [rbp-38h]

  v2 = a1 + 1;
  v5 = (_QWORD *)a1[2];
  if ( v5 )
  {
    v6 = *a2;
    v7 = a1 + 1;
    do
    {
      while ( 1 )
      {
        v8 = v5[2];
        v9 = v5[3];
        if ( v5[4] >= (unsigned __int64)v6 )
          break;
        v5 = (_QWORD *)v5[3];
        if ( !v9 )
          goto LABEL_6;
      }
      v7 = v5;
      v5 = (_QWORD *)v5[2];
    }
    while ( v8 );
LABEL_6:
    if ( v2 != v7 && v7[4] <= (unsigned __int64)v6 )
    {
LABEL_8:
      result = (char *)*((unsigned int *)v7 + 10);
      if ( (_DWORD)result )
        return result;
      goto LABEL_16;
    }
  }
  else
  {
    v7 = a1 + 1;
  }
  v11 = sub_22077B0(48);
  v12 = *a2;
  v13 = v7;
  *(_DWORD *)(v11 + 40) = 0;
  v7 = (_QWORD *)v11;
  *(_QWORD *)(v11 + 32) = v12;
  v19 = v12;
  v14 = sub_17BD040(a1, v13, (unsigned __int64 *)(v11 + 32));
  if ( !v15 )
  {
    v18 = v7;
    v7 = v14;
    j_j___libc_free_0(v18, 48);
    goto LABEL_8;
  }
  v16 = v14 || v2 == v15 || (unsigned __int64)v19 < v15[4];
  sub_220F040(v16, v7, v15, v2);
  ++a1[5];
  result = (char *)*((unsigned int *)v7 + 10);
  if ( !(_DWORD)result )
  {
LABEL_16:
    result = (char *)((unsigned int)((__int64)(a1[7] - a1[6]) >> 3) + 1);
    *((_DWORD *)v7 + 10) = (_DWORD)result;
    v17 = (_BYTE *)a1[7];
    if ( v17 == (_BYTE *)a1[8] )
    {
      return sub_1292090((__int64)(a1 + 6), v17, a2);
    }
    else
    {
      if ( v17 )
      {
        result = *a2;
        *(_QWORD *)v17 = *a2;
        v17 = (_BYTE *)a1[7];
      }
      a1[7] = v17 + 8;
    }
  }
  return result;
}
