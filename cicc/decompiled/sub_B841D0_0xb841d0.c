// Function: sub_B841D0
// Address: 0xb841d0
//
char *__fastcall sub_B841D0(char **a1, __int64 a2)
{
  __int64 v2; // rbx
  char *result; // rax
  __int64 v4; // r13
  __int64 v5; // rax
  char *v6; // rsi
  __int64 v7[5]; // [rsp+8h] [rbp-28h] BYREF

  v2 = a2;
  v7[0] = a2;
  result = a1[1];
  if ( *a1 != result )
  {
    v4 = *(_QWORD *)(*((_QWORD *)result - 1) + 8LL);
    v5 = *(unsigned int *)(v4 + 120);
    if ( v5 + 1 > (unsigned __int64)*(unsigned int *)(v4 + 124) )
    {
      sub_C8D5F0(v4 + 112, v4 + 128, v5 + 1, 8);
      v5 = *(unsigned int *)(v4 + 120);
    }
    *(_QWORD *)(*(_QWORD *)(v4 + 112) + 8 * v5) = a2;
    v2 = v7[0];
    ++*(_DWORD *)(v4 + 120);
    *(_QWORD *)(v2 + 8) = v4;
    result = (char *)(unsigned int)(*(_DWORD *)(*((_QWORD *)a1[1] - 1) + 384LL) + 1);
    *(_DWORD *)(v2 + 384) = (_DWORD)result;
    v6 = a1[1];
    if ( v6 != a1[2] )
      goto LABEL_5;
    return sub_B84040((__int64)a1, v6, v7);
  }
  *(_DWORD *)(a2 + 384) = 1;
  v6 = a1[1];
  if ( v6 == a1[2] )
    return sub_B84040((__int64)a1, v6, v7);
LABEL_5:
  if ( v6 )
  {
    *(_QWORD *)v6 = v2;
    v6 = a1[1];
  }
  a1[1] = v6 + 8;
  return result;
}
