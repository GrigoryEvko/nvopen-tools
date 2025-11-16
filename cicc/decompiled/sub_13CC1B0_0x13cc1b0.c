// Function: sub_13CC1B0
// Address: 0x13cc1b0
//
unsigned __int64 __fastcall sub_13CC1B0(__int64 a1, __int64 a2)
{
  unsigned int v2; // ecx
  unsigned __int64 result; // rax

  v2 = *(_DWORD *)(a1 + 8);
  if ( v2 > 0x40 )
    return sub_16A7DC0(a1, a2);
  result = 0;
  if ( v2 != (_DWORD)a2 )
    result = (*(_QWORD *)a1 << a2) & (0xFFFFFFFFFFFFFFFFLL >> -(char)v2);
  *(_QWORD *)a1 = result;
  return result;
}
