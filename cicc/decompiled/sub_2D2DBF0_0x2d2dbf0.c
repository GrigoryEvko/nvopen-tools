// Function: sub_2D2DBF0
// Address: 0x2d2dbf0
//
__int64 __fastcall sub_2D2DBF0(_QWORD *a1, unsigned __int64 *a2)
{
  _QWORD *v2; // r14
  _QWORD *v5; // rax
  unsigned __int64 v6; // rsi
  __int64 v7; // r12
  __int64 v8; // rcx
  __int64 v9; // rdx
  __int64 result; // rax
  __int64 v11; // rax
  unsigned __int64 v12; // rcx
  __int64 v13; // rsi
  _QWORD *v14; // rax
  _QWORD *v15; // rdx
  char v16; // di
  char *v17; // rsi
  unsigned __int64 v18; // rdi
  unsigned __int64 v19; // [rsp+8h] [rbp-38h]

  v2 = a1 + 1;
  v5 = (_QWORD *)a1[2];
  if ( v5 )
  {
    v6 = *a2;
    v7 = (__int64)(a1 + 1);
    do
    {
      while ( 1 )
      {
        v8 = v5[2];
        v9 = v5[3];
        if ( v5[4] >= v6 )
          break;
        v5 = (_QWORD *)v5[3];
        if ( !v9 )
          goto LABEL_6;
      }
      v7 = (__int64)v5;
      v5 = (_QWORD *)v5[2];
    }
    while ( v8 );
LABEL_6:
    if ( v2 != (_QWORD *)v7 && v6 >= *(_QWORD *)(v7 + 32) )
    {
LABEL_8:
      result = *(unsigned int *)(v7 + 40);
      if ( (_DWORD)result )
        return result;
      goto LABEL_16;
    }
  }
  else
  {
    v7 = (__int64)(a1 + 1);
  }
  v11 = sub_22077B0(0x30u);
  v12 = *a2;
  v13 = v7;
  *(_DWORD *)(v11 + 40) = 0;
  v7 = v11;
  *(_QWORD *)(v11 + 32) = v12;
  v19 = v12;
  v14 = sub_2D2DAF0(a1, v13, (unsigned __int64 *)(v11 + 32));
  if ( !v15 )
  {
    v18 = v7;
    v7 = (__int64)v14;
    j_j___libc_free_0(v18);
    goto LABEL_8;
  }
  v16 = v14 || v2 == v15 || v19 < v15[4];
  sub_220F040(v16, v7, v15, v2);
  ++a1[5];
  result = *(unsigned int *)(v7 + 40);
  if ( !(_DWORD)result )
  {
LABEL_16:
    *(_DWORD *)(v7 + 40) = ((__int64)(a1[7] - a1[6]) >> 3) + 1;
    v17 = (char *)a1[7];
    if ( v17 == (char *)a1[8] )
    {
      sub_2D2AD20(a1 + 6, v17, a2);
    }
    else
    {
      if ( v17 )
      {
        *(_QWORD *)v17 = *a2;
        v17 = (char *)a1[7];
      }
      a1[7] = v17 + 8;
    }
    return *(unsigned int *)(v7 + 40);
  }
  return result;
}
