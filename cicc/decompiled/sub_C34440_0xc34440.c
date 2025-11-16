// Function: sub_C34440
// Address: 0xc34440
//
__int64 __fastcall sub_C34440(unsigned __int8 *a1)
{
  __int64 result; // rax

  result = a1[20];
  if ( *(_DWORD *)(*(_QWORD *)a1 + 20LL) != 2 || (result & 5) != 1 )
  {
    result = a1[20] & 0xF7 | ~(_BYTE)result & 8u;
    a1[20] = result;
  }
  return result;
}
