// Function: sub_135E0D0
// Address: 0x135e0d0
//
__int64 __fastcall sub_135E0D0(__int64 a1, unsigned int a2, __int64 a3, unsigned __int8 a4)
{
  __int64 result; // rax

  *(_DWORD *)(a1 + 8) = a2;
  if ( a2 > 0x40 )
    return sub_16A4EF0(a1, a3, a4);
  *(_QWORD *)a1 = a3 & (0xFFFFFFFFFFFFFFFFLL >> -(char)a2);
  return result;
}
