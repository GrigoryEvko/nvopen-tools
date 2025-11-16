// Function: sub_A74940
// Address: 0xa74940
//
__int64 __fastcall sub_A74940(__int64 a1, __int64 a2, __int64 a3)
{
  __int64 v3; // r13
  const void *v4; // r12
  __int64 v5; // rax
  size_t v6; // r8
  __int64 v7; // r14
  __int64 result; // rax
  __int64 v9[7]; // [rsp+8h] [rbp-38h] BYREF

  *(_QWORD *)a1 = a2;
  *(_QWORD *)(a1 + 8) = a1 + 24;
  *(_QWORD *)(a1 + 16) = 0x800000000LL;
  v9[0] = a3;
  v3 = sub_A73290(v9);
  v4 = (const void *)sub_A73280(v9);
  v5 = *(unsigned int *)(a1 + 16);
  v6 = v3 - (_QWORD)v4;
  v7 = (v3 - (__int64)v4) >> 3;
  if ( v5 + v7 > (unsigned __int64)*(unsigned int *)(a1 + 20) )
  {
    sub_C8D5F0(a1 + 8, a1 + 24, v5 + v7, 8);
    v5 = *(unsigned int *)(a1 + 16);
    v6 = v3 - (_QWORD)v4;
  }
  if ( (const void *)v3 != v4 )
  {
    memcpy((void *)(*(_QWORD *)(a1 + 8) + 8 * v5), v4, v6);
    v5 = *(unsigned int *)(a1 + 16);
  }
  result = v7 + v5;
  *(_DWORD *)(a1 + 16) = result;
  return result;
}
