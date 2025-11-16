// Function: sub_38701C0
// Address: 0x38701c0
//
__int64 __fastcall sub_38701C0(__int64 **a1, __int64 *a2, __int64 a3, __int64 a4, int a5, int a6)
{
  __int64 v7; // rax
  __int64 v8; // rsi
  __int64 result; // rax

  v7 = a2[1];
  *a1 = a2;
  a1[1] = (__int64 *)v7;
  a1[2] = (__int64 *)a2[2];
  v8 = *a2;
  a1[3] = (__int64 *)v8;
  if ( v8 )
    sub_1623A60((__int64)(a1 + 3), v8, 2);
  a1[4] = (__int64 *)a3;
  result = *(unsigned int *)(a3 + 344);
  if ( (unsigned int)result >= *(_DWORD *)(a3 + 348) )
  {
    sub_16CD150(a3 + 336, (const void *)(a3 + 352), 0, 8, a5, a6);
    result = *(unsigned int *)(a3 + 344);
  }
  *(_QWORD *)(*(_QWORD *)(a3 + 336) + 8 * result) = a1;
  ++*(_DWORD *)(a3 + 344);
  return result;
}
