// Function: sub_2E7D2B0
// Address: 0x2e7d2b0
//
__int64 __fastcall sub_2E7D2B0(unsigned __int64 *a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  __int64 v8; // rbx
  __int64 v9; // r8
  __int64 v10; // r9
  __int64 v11; // rax
  __int64 result; // rax
  unsigned __int64 v13; // rcx

  v8 = sub_2E7D0F0(a1, a2, a3, a4, a5, a6);
  v11 = *(unsigned int *)(v8 + 16);
  if ( v11 + 1 > (unsigned __int64)*(unsigned int *)(v8 + 20) )
  {
    sub_C8D5F0(v8 + 8, (const void *)(v8 + 24), v11 + 1, 8u, v9, v10);
    v11 = *(unsigned int *)(v8 + 16);
  }
  *(_QWORD *)(*(_QWORD *)(v8 + 8) + 8 * v11) = a3;
  result = *(unsigned int *)(v8 + 40);
  v13 = *(unsigned int *)(v8 + 44);
  ++*(_DWORD *)(v8 + 16);
  if ( result + 1 > v13 )
  {
    sub_C8D5F0(v8 + 32, (const void *)(v8 + 48), result + 1, 8u, v9, v10);
    result = *(unsigned int *)(v8 + 40);
  }
  *(_QWORD *)(*(_QWORD *)(v8 + 32) + 8 * result) = a4;
  ++*(_DWORD *)(v8 + 40);
  return result;
}
