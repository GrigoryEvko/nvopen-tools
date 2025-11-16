// Function: sub_FFA910
// Address: 0xffa910
//
__int64 __fastcall sub_FFA910(__int64 *a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  __int64 v6; // rbx
  __int64 result; // rax

  v6 = *a1;
  result = *(unsigned int *)(*a1 + 8);
  if ( result + 1 > (unsigned __int64)*(unsigned int *)(*a1 + 12) )
  {
    sub_C8D5F0(*a1, (const void *)(v6 + 16), result + 1, 8u, a5, a6);
    result = *(unsigned int *)(v6 + 8);
  }
  *(_QWORD *)(*(_QWORD *)v6 + 8 * result) = a2;
  ++*(_DWORD *)(v6 + 8);
  return result;
}
