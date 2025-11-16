// Function: sub_2043460
// Address: 0x2043460
//
char *__fastcall sub_2043460(_QWORD *a1, __int64 a2)
{
  __int64 v2; // r12
  _QWORD *v3; // rbx
  _QWORD *v4; // r12
  int v5; // r13d
  unsigned __int64 v6; // r8
  char *v7; // rax
  __int64 v8; // rdx
  char *result; // rax
  _BYTE *v10; // rsi
  char *v11; // [rsp+8h] [rbp-28h] BYREF

  v2 = *(unsigned int *)(a2 + 120);
  v3 = *(_QWORD **)(a2 + 112);
  v11 = (char *)a2;
  v4 = &v3[2 * v2];
  if ( v4 == v3 )
  {
    v7 = (char *)a2;
    v5 = 0;
  }
  else
  {
    v5 = 0;
    do
    {
      v6 = sub_2042140((__int64)a1, *v3 & 0xFFFFFFFFFFFFFFF8LL);
      v7 = v11;
      v3 += 2;
      v5 += v11 == (char *)v6;
    }
    while ( v4 != v3 );
  }
  v8 = *((unsigned int *)v7 + 48);
  result = (char *)a1[3];
  *(_DWORD *)&result[4 * v8] = v5;
  v10 = (_BYTE *)a1[7];
  if ( v10 == (_BYTE *)a1[8] )
    return sub_1CFD630((__int64)(a1 + 6), v10, &v11);
  if ( v10 )
  {
    result = v11;
    *(_QWORD *)v10 = v11;
    v10 = (_BYTE *)a1[7];
  }
  a1[7] = v10 + 8;
  return result;
}
