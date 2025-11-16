// Function: sub_16C93D0
// Address: 0x16c93d0
//
__int64 __fastcall sub_16C93D0(__int64 a1, __int64 *a2)
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
