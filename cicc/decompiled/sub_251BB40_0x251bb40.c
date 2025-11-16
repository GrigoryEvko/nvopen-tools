// Function: sub_251BB40
// Address: 0x251bb40
//
unsigned __int64 __fastcall sub_251BB40(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  __int64 **v6; // rax
  __int64 *v7; // rbx
  unsigned __int64 result; // rax
  __int64 *i; // r12
  __int64 v10; // rax
  __int64 v11; // rdx
  unsigned __int64 v12[5]; // [rsp+8h] [rbp-28h] BYREF

  v6 = *(__int64 ***)(*(_QWORD *)(a1 + 400) + 8LL * *(unsigned int *)(a1 + 408) - 8);
  v7 = *v6;
  result = 3LL * *((unsigned int *)v6 + 2);
  for ( i = &v7[result]; i != v7; result = sub_251B630(v10 + 8, v12, v11, a4, a5, a6) )
  {
    v10 = *v7;
    v11 = *((unsigned int *)v7 + 4);
    v7 += 3;
    v11 *= 4;
    v12[0] = v11 | *(v7 - 2) & 0xFFFFFFFFFFFFFFFBLL;
  }
  return result;
}
