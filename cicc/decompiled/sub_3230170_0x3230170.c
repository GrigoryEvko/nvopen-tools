// Function: sub_3230170
// Address: 0x3230170
//
_QWORD *__fastcall sub_3230170(__int64 a1, unsigned int a2, _QWORD *a3, __int64 a4, __int64 a5, __int64 a6)
{
  __int64 *v6; // r15
  __int64 *v8; // r12
  __int64 v9; // r14
  _QWORD *result; // rax
  __int64 v11; // rbx
  __int64 v12; // r8
  __int64 v13; // r9
  __int64 v14; // rax
  __int64 v15; // [rsp+0h] [rbp-50h]
  __int64 v16[7]; // [rsp+18h] [rbp-38h] BYREF

  v6 = (__int64 *)a4;
  v8 = (__int64 *)(a4 + 16 * a5);
  v16[0] = a2;
  v9 = sub_322FCF0(a1, v16, (__int64)a3, a4, a5, a6);
  for ( result = (_QWORD *)(v9 + 16); v8 != v6; ++*(_DWORD *)(v9 + 8) )
  {
    v11 = *v6;
    v12 = sub_32237B0(a3, v6[1]);
    v14 = *(unsigned int *)(v9 + 8);
    if ( v14 + 1 > (unsigned __int64)*(unsigned int *)(v9 + 12) )
    {
      v15 = v12;
      sub_C8D5F0(v9, (const void *)(v9 + 16), v14 + 1, 0x10u, v12, v13);
      v14 = *(unsigned int *)(v9 + 8);
      v12 = v15;
    }
    v6 += 2;
    result = (_QWORD *)(*(_QWORD *)v9 + 16 * v14);
    *result = v11;
    result[1] = v12;
  }
  return result;
}
