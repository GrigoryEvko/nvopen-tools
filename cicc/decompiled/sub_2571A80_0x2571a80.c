// Function: sub_2571A80
// Address: 0x2571a80
//
__int64 __fastcall sub_2571A80(__int64 a1, __int64 *a2)
{
  _QWORD *v4; // rdi
  _QWORD *v6; // rsi
  __int64 v7; // rdx
  __int64 v8; // rcx
  __int64 v9; // r8
  __int64 v10; // r9
  __int64 result; // rax
  __int64 v12; // r9
  __int64 v13; // rdx
  __int64 v14; // rbx
  unsigned __int8 v15; // [rsp+Fh] [rbp-41h]
  _BYTE v16[32]; // [rsp+10h] [rbp-40h] BYREF
  unsigned __int8 v17; // [rsp+30h] [rbp-20h]

  if ( *(_DWORD *)(a1 + 16) )
  {
    sub_2571760((__int64)v16, a1, a2);
    result = v17;
    if ( !v17 )
      return 0;
    v13 = *(unsigned int *)(a1 + 40);
    v14 = *a2;
    if ( v13 + 1 > (unsigned __int64)*(unsigned int *)(a1 + 44) )
    {
      v15 = v17;
      sub_C8D5F0(a1 + 32, (const void *)(a1 + 48), v13 + 1, 8u, v13 + 1, v12);
      v13 = *(unsigned int *)(a1 + 40);
      result = v15;
    }
    *(_QWORD *)(*(_QWORD *)(a1 + 32) + 8 * v13) = v14;
    ++*(_DWORD *)(a1 + 40);
  }
  else
  {
    v4 = *(_QWORD **)(a1 + 32);
    v6 = &v4[*(unsigned int *)(a1 + 40)];
    if ( v6 != sub_2538080(v4, (__int64)v6, a2) )
      return 0;
    return sub_25718D0(a1, *a2, v7, v8, v9, v10);
  }
  return result;
}
