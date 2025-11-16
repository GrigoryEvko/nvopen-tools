// Function: sub_1636A40
// Address: 0x1636a40
//
__int64 __fastcall sub_1636A40(__int64 a1, __int64 a2)
{
  __int64 result; // rax
  __int64 v3; // rdx

  result = a1;
  v3 = *(unsigned int *)(a1 + 8);
  if ( (unsigned int)v3 >= *(_DWORD *)(a1 + 12) )
  {
    sub_16CD150(a1, a1 + 16, 0, 8);
    result = a1;
    v3 = *(unsigned int *)(a1 + 8);
  }
  *(_QWORD *)(*(_QWORD *)result + 8 * v3) = a2;
  ++*(_DWORD *)(result + 8);
  return result;
}
