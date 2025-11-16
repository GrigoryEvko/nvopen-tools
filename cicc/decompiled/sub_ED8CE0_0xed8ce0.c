// Function: sub_ED8CE0
// Address: 0xed8ce0
//
unsigned __int64 __fastcall sub_ED8CE0(__int64 a1, __int64 a2)
{
  const void *v2; // r13
  unsigned __int64 result; // rax
  char *v5; // r14
  signed __int64 v6; // rdx
  signed __int64 v7; // r15
  __int64 v8; // rsi

  if ( a2 < 0 )
    sub_4262D8((__int64)"vector::reserve");
  v2 = *(const void **)a1;
  result = *(_QWORD *)(a1 + 16) - *(_QWORD *)a1;
  if ( result < a2 )
  {
    v5 = 0;
    v6 = *(_QWORD *)(a1 + 8) - (_QWORD)v2;
    v7 = v6;
    if ( a2 )
    {
      result = sub_22077B0(a2);
      v2 = *(const void **)a1;
      v5 = (char *)result;
      v6 = *(_QWORD *)(a1 + 8) - *(_QWORD *)a1;
    }
    if ( v6 > 0 )
    {
      memmove(v5, v2, v6);
      v8 = *(_QWORD *)(a1 + 16) - (_QWORD)v2;
    }
    else
    {
      if ( !v2 )
      {
LABEL_7:
        *(_QWORD *)a1 = v5;
        *(_QWORD *)(a1 + 8) = &v5[v7];
        *(_QWORD *)(a1 + 16) = &v5[a2];
        return result;
      }
      v8 = *(_QWORD *)(a1 + 16) - (_QWORD)v2;
    }
    result = j_j___libc_free_0(v2, v8);
    goto LABEL_7;
  }
  return result;
}
