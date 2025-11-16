// Function: sub_C88FD0
// Address: 0xc88fd0
//
__int64 __fastcall sub_C88FD0(__int64 a1, __int64 *a2)
{
  __int64 v2; // rax
  __int64 result; // rax

  v2 = *a2;
  *a2 = 0;
  *(_QWORD *)a1 = v2;
  result = *((unsigned int *)a2 + 2);
  *((_DWORD *)a2 + 2) = 2;
  *(_DWORD *)(a1 + 8) = result;
  return result;
}
