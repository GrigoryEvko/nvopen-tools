// Function: sub_B6B1A0
// Address: 0xb6b1a0
//
__int64 __fastcall sub_B6B1A0(__int64 a1, __int64 a2, int a3)
{
  __int64 v3; // rax
  __int64 result; // rax

  v3 = *(unsigned int *)(a1 + 8);
  if ( v3 + 1 > (unsigned __int64)*(unsigned int *)(a1 + 12) )
  {
    sub_C8D5F0(a1, a1 + 16, v3 + 1, 12);
    v3 = *(unsigned int *)(a1 + 8);
  }
  result = *(_QWORD *)a1 + 12 * v3;
  *(_QWORD *)result = a2;
  *(_DWORD *)(result + 8) = a3;
  ++*(_DWORD *)(a1 + 8);
  return result;
}
