// Function: sub_A18B40
// Address: 0xa18b40
//
__int64 __fastcall sub_A18B40(__int64 a1, _BYTE *a2, _BYTE *a3)
{
  size_t v3; // r15
  __int64 v5; // r13
  __int64 result; // rax

  v3 = a3 - a2;
  v5 = (a3 - a2) >> 2;
  result = *(unsigned int *)(a1 + 8);
  if ( v5 + result > (unsigned __int64)*(unsigned int *)(a1 + 12) )
  {
    sub_C8D5F0(a1, a1 + 16, v5 + result, 4);
    result = *(unsigned int *)(a1 + 8);
  }
  if ( a2 != a3 )
  {
    memcpy((void *)(*(_QWORD *)a1 + 4 * result), a2, v3);
    result = *(unsigned int *)(a1 + 8);
  }
  *(_DWORD *)(a1 + 8) = result + v5;
  return result;
}
