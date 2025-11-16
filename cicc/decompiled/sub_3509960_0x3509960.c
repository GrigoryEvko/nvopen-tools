// Function: sub_3509960
// Address: 0x3509960
//
__int64 __fastcall sub_3509960(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  __int64 v7; // rdi
  __int64 v8; // rbx
  __int64 result; // rax

  v7 = *(_QWORD *)(a1 + 40);
  if ( v7 )
    sub_300BAC0(v7, a2, a3, a4, a5, a6);
  v8 = *(_QWORD *)(a1 + 16);
  result = *(unsigned int *)(v8 + 8);
  if ( result + 1 > (unsigned __int64)*(unsigned int *)(v8 + 12) )
  {
    sub_C8D5F0(v8, (const void *)(v8 + 16), result + 1, 4u, a5, a6);
    result = *(unsigned int *)(v8 + 8);
  }
  *(_DWORD *)(*(_QWORD *)v8 + 4 * result) = a2;
  ++*(_DWORD *)(v8 + 8);
  return result;
}
