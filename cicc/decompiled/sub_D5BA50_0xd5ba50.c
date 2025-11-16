// Function: sub_D5BA50
// Address: 0xd5ba50
//
__int64 *__fastcall sub_D5BA50(__int64 **a1, __int64 *a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  __int64 *result; // rax
  __int64 v7; // rbx
  __int64 v8; // r12

  result = *a1;
  v7 = **a1;
  if ( v7 )
  {
    result = (__int64 *)*(unsigned int *)(v7 + 8);
    v8 = *a2;
    if ( (unsigned __int64)result + 1 > *(unsigned int *)(v7 + 12) )
    {
      sub_C8D5F0(**a1, (const void *)(v7 + 16), (unsigned __int64)result + 1, 8u, a5, a6);
      result = (__int64 *)*(unsigned int *)(v7 + 8);
    }
    *(_QWORD *)(*(_QWORD *)v7 + 8LL * (_QWORD)result) = v8;
    ++*(_DWORD *)(v7 + 8);
  }
  return result;
}
