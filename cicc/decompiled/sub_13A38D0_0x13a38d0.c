// Function: sub_13A38D0
// Address: 0x13a38d0
//
__int64 __fastcall sub_13A38D0(__int64 a1, __int64 a2)
{
  unsigned int v2; // eax
  __int64 result; // rax

  v2 = *(_DWORD *)(a2 + 8);
  *(_DWORD *)(a1 + 8) = v2;
  if ( v2 > 0x40 )
    return sub_16A4FD0(a1, a2);
  result = *(_QWORD *)a2;
  *(_QWORD *)a1 = *(_QWORD *)a2;
  return result;
}
