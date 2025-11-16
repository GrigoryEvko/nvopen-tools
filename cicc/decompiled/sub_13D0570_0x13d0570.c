// Function: sub_13D0570
// Address: 0x13d0570
//
unsigned __int64 __fastcall sub_13D0570(__int64 a1)
{
  unsigned int v1; // ecx
  unsigned __int64 result; // rax

  v1 = *(_DWORD *)(a1 + 8);
  if ( v1 > 0x40 )
    return sub_16A8F40(a1);
  result = ~*(_QWORD *)a1 & (0xFFFFFFFFFFFFFFFFLL >> -(char)v1);
  *(_QWORD *)a1 = result;
  return result;
}
