// Function: sub_16A7C60
// Address: 0x16a7c60
//
void *__fastcall sub_16A7C60(_QWORD *a1, unsigned __int64 *a2, __int64 a3, unsigned int a4, unsigned int a5)
{
  _QWORD *v6; // r14
  unsigned int v10; // eax
  unsigned __int64 *v11; // rax
  void *result; // rax
  unsigned int v13; // r9d
  __int64 v14; // r13
  __int64 v15; // rdi
  unsigned __int64 v16; // rdx
  void *v17; // [rsp-10h] [rbp-50h]
  unsigned int v18; // [rsp+Ch] [rbp-34h]

  v6 = a1;
  while ( a4 > a5 )
  {
    v10 = a4;
    a4 = a5;
    a5 = v10;
    v11 = a2;
    a2 = (unsigned __int64 *)a3;
    a3 = (__int64)v11;
  }
  result = sub_16A7020(a1, 0, a5);
  if ( a4 )
  {
    v13 = a5 + 1;
    v14 = (__int64)&a2[a4];
    do
    {
      v15 = (__int64)v6;
      v16 = *a2++;
      v18 = v13;
      ++v6;
      sub_16A7890(v15, a3, v16, 0, a5, v13, 1);
      result = v17;
      v13 = v18;
    }
    while ( a2 != (unsigned __int64 *)v14 );
  }
  return result;
}
