// Function: sub_3111E70
// Address: 0x3111e70
//
__int64 __fastcall sub_3111E70(
        unsigned __int8 *src,
        size_t n,
        unsigned __int8 *a3,
        size_t a4,
        __int64 a5,
        __int64 a6,
        char a7)
{
  char *v10; // rdi
  char *v11; // r12
  char *v12; // rsi
  __int64 result; // rax
  _QWORD *v14; // rax
  void *v17; // rdi
  __int64 v18; // rbx
  __int64 v19; // rax
  _WORD *v20; // r12
  __int64 v21; // rax
  void *v22; // rdi
  __int64 v23; // r12
  _BYTE *v24; // r13

  sub_CA5BD0((__int64)src, n);
  if ( !n )
    goto LABEL_2;
  v14 = sub_CB72A0();
  v17 = (void *)v14[4];
  v18 = (__int64)v14;
  if ( v14[3] - (_QWORD)v17 < n )
  {
    v18 = sub_CB6200((__int64)v14, src, n);
    v20 = *(_WORD **)(v18 + 32);
    if ( *(_QWORD *)(v18 + 24) - (_QWORD)v20 > 1u )
      goto LABEL_8;
  }
  else
  {
    memcpy(v17, src, n);
    v19 = *(_QWORD *)(v18 + 24);
    v20 = (_WORD *)(*(_QWORD *)(v18 + 32) + n);
    *(_QWORD *)(v18 + 32) = v20;
    if ( (unsigned __int64)(v19 - (_QWORD)v20) > 1 )
    {
LABEL_8:
      *v20 = 8250;
      *(_QWORD *)(v18 + 32) += 2LL;
      goto LABEL_2;
    }
  }
  sub_CB6200(v18, (unsigned __int8 *)": ", 2u);
LABEL_2:
  v10 = &a7;
  v11 = (char *)sub_CB72A0();
  v12 = v11;
  sub_CA0E80((__int64)&a7, (__int64)v11);
  result = *((_QWORD *)v11 + 4);
  if ( *((_QWORD *)v11 + 3) == result )
  {
    v12 = "\n";
    v10 = v11;
    result = sub_CB6200((__int64)v11, (unsigned __int8 *)"\n", 1u);
  }
  else
  {
    *(_BYTE *)result = 10;
    ++*((_QWORD *)v11 + 4);
  }
  if ( a4 )
  {
    v21 = sub_CA5D20((__int64)v10, (__int64)v12);
    v22 = *(void **)(v21 + 32);
    v23 = v21;
    if ( *(_QWORD *)(v21 + 24) - (_QWORD)v22 < a4 )
    {
      result = sub_CB6200(v21, a3, a4);
      v24 = *(_BYTE **)(result + 32);
      v23 = result;
    }
    else
    {
      result = (__int64)memcpy(v22, a3, a4);
      v24 = (_BYTE *)(*(_QWORD *)(v23 + 32) + a4);
      *(_QWORD *)(v23 + 32) = v24;
    }
    if ( *(_BYTE **)(v23 + 24) == v24 )
    {
      return sub_CB6200(v23, (unsigned __int8 *)"\n", 1u);
    }
    else
    {
      *v24 = 10;
      ++*(_QWORD *)(v23 + 32);
    }
  }
  return result;
}
