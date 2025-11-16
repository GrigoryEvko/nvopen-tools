// Function: sub_B90330
// Address: 0xb90330
//
__int64 __fastcall sub_B90330(__int64 a1, __int64 a2, __int64 a3)
{
  __int64 v4; // rax
  __int64 result; // rax
  unsigned __int64 v6; // rcx

  v4 = *(unsigned int *)(a1 + 8);
  if ( (_DWORD)v4 )
  {
    result = sub_B8FF20(a1, a2, a3);
    if ( (_BYTE)result )
      return result;
    v4 = *(unsigned int *)(a1 + 8);
  }
  if ( v4 + 1 > (unsigned __int64)*(unsigned int *)(a1 + 12) )
  {
    sub_C8D5F0(a1, a1 + 16, v4 + 1, 8);
    v4 = *(unsigned int *)(a1 + 8);
  }
  *(_QWORD *)(*(_QWORD *)a1 + 8 * v4) = a2;
  v6 = *(unsigned int *)(a1 + 12);
  result = (unsigned int)(*(_DWORD *)(a1 + 8) + 1);
  *(_DWORD *)(a1 + 8) = result;
  if ( result + 1 > v6 )
  {
    sub_C8D5F0(a1, a1 + 16, result + 1, 8);
    result = *(unsigned int *)(a1 + 8);
  }
  *(_QWORD *)(*(_QWORD *)a1 + 8 * result) = a3;
  ++*(_DWORD *)(a1 + 8);
  return result;
}
