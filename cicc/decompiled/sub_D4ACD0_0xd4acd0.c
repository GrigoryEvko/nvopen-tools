// Function: sub_D4ACD0
// Address: 0xd4acd0
//
unsigned __int64 __fastcall sub_D4ACD0(__int64 a1, unsigned __int64 a2)
{
  const void *v2; // r12
  unsigned __int64 result; // rax
  __int64 v4; // r14
  __int64 v5; // r13
  char *v6; // r15
  signed __int64 v7; // rdx
  __int64 v8; // rsi

  if ( a2 > 0xFFFFFFFFFFFFFFFLL )
    sub_4262D8((__int64)"vector::reserve");
  v2 = *(const void **)a1;
  result = (__int64)(*(_QWORD *)(a1 + 16) - *(_QWORD *)a1) >> 3;
  if ( a2 > result )
  {
    v4 = 8 * a2;
    v5 = *(_QWORD *)(a1 + 8) - (_QWORD)v2;
    if ( a2 )
    {
      result = sub_22077B0(8 * a2);
      v2 = *(const void **)a1;
      v6 = (char *)result;
      v7 = *(_QWORD *)(a1 + 8) - *(_QWORD *)a1;
      if ( v7 <= 0 )
      {
LABEL_5:
        if ( !v2 )
        {
LABEL_6:
          *(_QWORD *)a1 = v6;
          *(_QWORD *)(a1 + 8) = &v6[v5];
          *(_QWORD *)(a1 + 16) = &v6[v4];
          return result;
        }
        v8 = *(_QWORD *)(a1 + 16) - (_QWORD)v2;
LABEL_10:
        result = j_j___libc_free_0(v2, v8);
        goto LABEL_6;
      }
    }
    else
    {
      v7 = *(_QWORD *)(a1 + 8) - (_QWORD)v2;
      v6 = 0;
      if ( v5 <= 0 )
        goto LABEL_5;
    }
    memmove(v6, v2, v7);
    v8 = *(_QWORD *)(a1 + 16) - (_QWORD)v2;
    goto LABEL_10;
  }
  return result;
}
