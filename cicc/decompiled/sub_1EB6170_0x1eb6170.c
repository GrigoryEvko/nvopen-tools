// Function: sub_1EB6170
// Address: 0x1eb6170
//
__int64 __fastcall sub_1EB6170(_QWORD *a1, int a2, __int64 a3, __int64 a4, __int64 a5, int a6)
{
  __int64 v6; // rdx
  __int64 result; // rax

  v6 = a2 & 0x7FFFFFFF;
  result = *(unsigned int *)(*(_QWORD *)(a1[32] + 264LL) + 4 * v6);
  if ( (_DWORD)result )
    return sub_1EB5F30(a1, a2, v6, a4, a5, a6);
  return result;
}
