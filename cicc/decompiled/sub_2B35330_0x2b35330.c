// Function: sub_2B35330
// Address: 0x2b35330
//
__int64 __fastcall sub_2B35330(__int64 a1, _BYTE *a2, _BYTE *a3, __int64 a4, __int64 a5, __int64 a6)
{
  size_t v6; // r15
  __int64 v8; // r13
  __int64 result; // rax

  v6 = a3 - a2;
  v8 = (a3 - a2) >> 2;
  result = *(unsigned int *)(a1 + 8);
  if ( v8 + result > (unsigned __int64)*(unsigned int *)(a1 + 12) )
  {
    sub_C8D5F0(a1, (const void *)(a1 + 16), v8 + result, 4u, a5, a6);
    result = *(unsigned int *)(a1 + 8);
  }
  if ( a2 != a3 )
  {
    memcpy((void *)(*(_QWORD *)a1 + 4 * result), a2, v6);
    result = *(unsigned int *)(a1 + 8);
  }
  *(_DWORD *)(a1 + 8) = result + v8;
  return result;
}
