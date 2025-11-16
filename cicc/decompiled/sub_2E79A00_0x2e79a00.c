// Function: sub_2E79A00
// Address: 0x2e79a00
//
char *__fastcall sub_2E79A00(__int64 a1, const void **a2, __int64 a3)
{
  unsigned __int64 v4; // r12
  char *v5; // rcx
  char *result; // rax
  const void *v7; // rsi
  signed __int64 v8; // r12

  v4 = (_BYTE *)a2[1] - (_BYTE *)*a2;
  *(_QWORD *)a1 = 0;
  *(_QWORD *)(a1 + 8) = 0;
  *(_QWORD *)(a1 + 16) = 0;
  if ( v4 )
  {
    if ( v4 > 0x7FFFFFFFFFFFFFF8LL )
      sub_4261EA(a1, a2, a3);
    v5 = (char *)sub_22077B0(v4);
  }
  else
  {
    v4 = 0;
    v5 = 0;
  }
  *(_QWORD *)a1 = v5;
  *(_QWORD *)(a1 + 16) = &v5[v4];
  *(_QWORD *)(a1 + 8) = v5;
  result = (char *)a2[1];
  v7 = *a2;
  v8 = (_BYTE *)a2[1] - (_BYTE *)*a2;
  if ( result != *a2 )
  {
    result = (char *)memmove(v5, v7, (_BYTE *)a2[1] - (_BYTE *)*a2);
    v5 = result;
  }
  *(_DWORD *)(a1 + 24) = 0;
  *(_QWORD *)(a1 + 8) = &v5[v8];
  return result;
}
