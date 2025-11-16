// Function: sub_EA2040
// Address: 0xea2040
//
__int64 __fastcall sub_EA2040(_QWORD *a1, __int64 a2, __int64 a3)
{
  __int64 v3; // rdx
  __int64 v4; // rcx
  __int64 v5; // r8
  __int64 v6; // r9
  __int64 v7; // rax
  unsigned __int8 **v8; // rbx
  __int64 result; // rax
  unsigned __int8 **i; // r13
  unsigned __int8 *v11; // rsi

  sub_E8B980(a1, a2, a3);
  v7 = a1[36];
  v8 = *(unsigned __int8 ***)(v7 + 72);
  result = 3LL * *(unsigned int *)(v7 + 80);
  for ( i = &v8[result]; i != v8; result = sub_EA1B60((__int64)a1, v11, v3, v4, v5, v6) )
  {
    v11 = *v8;
    v8 += 3;
  }
  return result;
}
