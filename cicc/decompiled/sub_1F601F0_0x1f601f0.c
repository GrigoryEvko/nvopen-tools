// Function: sub_1F601F0
// Address: 0x1f601f0
//
__int64 *__fastcall sub_1F601F0(__int64 a1, __int64 a2, __int64 a3, __int64 a4)
{
  unsigned __int64 v6; // rax
  __int64 v7; // rdx
  unsigned __int64 v8; // r12
  __int64 *result; // rax
  int v10; // r8d
  int v11; // r9d
  __int64 v12; // rax

  v6 = (unsigned int)*(unsigned __int8 *)(sub_157ED20(a1) + 16) - 34;
  if ( (unsigned int)v6 <= 0x36
    && (v7 = 0x40018000000001LL, _bittest64(&v7, v6))
    && (unsigned int)*(unsigned __int8 *)(sub_157ED20(a1) + 16) - 25 <= 9 )
  {
    v12 = *(unsigned int *)(a4 + 8);
    if ( (unsigned int)v12 >= *(_DWORD *)(a4 + 12) )
    {
      sub_16CD150(a4, (const void *)(a4 + 16), 0, 16, v10, v11);
      v12 = *(unsigned int *)(a4 + 8);
    }
    result = (__int64 *)(*(_QWORD *)a4 + 16 * v12);
    *result = a1;
    result[1] = a2;
    ++*(_DWORD *)(a4 + 8);
  }
  else
  {
    v8 = sub_157EBA0(a1);
    result = sub_1648A60(64, 2u);
    if ( result )
      return (__int64 *)sub_15F9660((__int64)result, a2, a3, v8);
  }
  return result;
}
