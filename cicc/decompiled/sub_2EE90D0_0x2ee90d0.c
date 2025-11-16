// Function: sub_2EE90D0
// Address: 0x2ee90d0
//
__int64 __fastcall sub_2EE90D0(__int64 a1, __int64 a2, unsigned int a3, __int64 a4, __int64 a5)
{
  __int64 v5; // r12
  __int64 v6; // r15
  __int64 result; // rax
  __int64 i; // r9
  __int64 v10; // r8
  __int64 v11; // rbx
  __int64 v12; // [rsp+0h] [rbp-40h]
  unsigned int v13; // [rsp+Ch] [rbp-34h]

  v5 = a4 + 8 * a5;
  v6 = *(_QWORD *)(a2 + 24);
  result = *(_QWORD *)(a2 + 32) + 40LL * a3;
  for ( i = *(unsigned int *)(result + 8); a4 != v5; ++*(_DWORD *)(v11 + 48) )
  {
    result = *(_QWORD *)(v5 - 8);
    if ( result == v6 )
      break;
    v10 = (unsigned int)i;
    v11 = *(_QWORD *)(a1 + 8) + 88LL * *(int *)(result + 24);
    result = *(unsigned int *)(v11 + 48);
    if ( result + 1 > (unsigned __int64)*(unsigned int *)(v11 + 52) )
    {
      v12 = (unsigned int)i;
      v13 = i;
      sub_C8D5F0(v11 + 40, (const void *)(v11 + 56), result + 1, 8u, (unsigned int)i, i);
      result = *(unsigned int *)(v11 + 48);
      v10 = v12;
      i = v13;
    }
    v5 -= 8;
    *(_QWORD *)(*(_QWORD *)(v11 + 40) + 8 * result) = v10;
  }
  return result;
}
