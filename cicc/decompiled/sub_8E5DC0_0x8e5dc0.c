// Function: sub_8E5DC0
// Address: 0x8e5dc0
//
unsigned __int64 __fastcall sub_8E5DC0(__int64 a1, int a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  int v6; // r14d
  __int64 v10; // r15
  __int64 v11; // rsi
  char *v12; // rdi
  unsigned __int64 result; // rax
  __int64 v14; // rsi
  char *v15; // rdi

  v6 = a4;
  v10 = qword_4F605B8;
  v11 = qword_4F605B0;
  v12 = (char *)qword_4F605C0;
  result = qword_4F605B8 + 1;
  qword_4F605B8 = result;
  if ( result <= qword_4F605B0
    || ((qword_4F605B0 += 500, v14 = 32 * (v11 + 500), !qword_4F605C0)
      ? (result = malloc(v14, v14, a3, a4, a5, a6))
      : (result = realloc(qword_4F605C0)),
        qword_4F605C0 = (void *)result,
        (v12 = (char *)result) != 0) )
  {
    v15 = &v12[32 * v10];
    *(_QWORD *)v15 = a1;
    *((_DWORD *)v15 + 2) = a2;
    *((_QWORD *)v15 + 2) = a3;
    *((_DWORD *)v15 + 6) = v6;
  }
  else
  {
    result = *(unsigned int *)(a5 + 24);
    if ( !(_DWORD)result )
    {
      ++*(_QWORD *)(a5 + 32);
      ++*(_QWORD *)(a5 + 48);
      *(_DWORD *)(a5 + 24) = 1;
    }
  }
  return result;
}
