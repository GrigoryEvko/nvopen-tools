// Function: sub_A718C0
// Address: 0xa718c0
//
__int64 __fastcall sub_A718C0(unsigned __int64 *a1, __int64 a2)
{
  __int64 v2; // rax
  unsigned __int64 v3; // r12
  unsigned __int64 v4; // r12
  unsigned __int64 v5; // rcx
  __int64 result; // rax

  v2 = *(unsigned int *)(a2 + 8);
  v3 = *a1;
  if ( v2 + 1 > (unsigned __int64)*(unsigned int *)(a2 + 12) )
  {
    sub_C8D5F0(a2, a2 + 16, v2 + 1, 4);
    v2 = *(unsigned int *)(a2 + 8);
  }
  *(_DWORD *)(*(_QWORD *)a2 + 4 * v2) = v3;
  v4 = HIDWORD(v3);
  v5 = *(unsigned int *)(a2 + 12);
  result = (unsigned int)(*(_DWORD *)(a2 + 8) + 1);
  *(_DWORD *)(a2 + 8) = result;
  if ( result + 1 > v5 )
  {
    sub_C8D5F0(a2, a2 + 16, result + 1, 4);
    result = *(unsigned int *)(a2 + 8);
  }
  *(_DWORD *)(*(_QWORD *)a2 + 4 * result) = v4;
  ++*(_DWORD *)(a2 + 8);
  return result;
}
