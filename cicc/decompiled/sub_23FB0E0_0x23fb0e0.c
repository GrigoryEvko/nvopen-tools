// Function: sub_23FB0E0
// Address: 0x23fb0e0
//
__int64 __fastcall sub_23FB0E0(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  char v6; // dl
  __int64 result; // rax

  *(_QWORD *)a1 = a1 + 16;
  *(_QWORD *)(a1 + 784) = a1 + 800;
  *(_QWORD *)(a1 + 8) = 0x800000000LL;
  *(_QWORD *)(a1 + 792) = 0x800000000LL;
  *(_QWORD *)(a1 + 944) = 0x800000000LL;
  *(_QWORD *)(a1 + 864) = 0;
  *(_QWORD *)(a1 + 872) = 0;
  *(_QWORD *)(a1 + 880) = 0;
  *(_QWORD *)(a1 + 888) = 0;
  *(_DWORD *)(a1 + 896) = 0;
  *(_QWORD *)(a1 + 904) = 0;
  *(_QWORD *)(a1 + 912) = 0;
  *(_QWORD *)(a1 + 920) = 0;
  *(_DWORD *)(a1 + 928) = 0;
  *(_QWORD *)(a1 + 936) = a1 + 952;
  *(_QWORD *)(a1 + 1720) = 0;
  *(_QWORD *)(a1 + 1728) = 0;
  *(_QWORD *)(a1 + 1736) = 0;
  *(_DWORD *)(a1 + 1744) = 0;
  *(_QWORD *)(a1 + 1752) = 0;
  *(_QWORD *)(a1 + 1760) = 0;
  *(_QWORD *)(a1 + 1768) = 0;
  *(_DWORD *)(a1 + 1776) = 0;
  *(_QWORD *)(a1 + 1784) = 0;
  *(_QWORD *)(a1 + 1792) = 0;
  *(_QWORD *)(a1 + 1800) = 0;
  *(_DWORD *)(a1 + 1808) = 0;
  *(_QWORD *)(a1 + 16) = *(_QWORD *)a2;
  v6 = *(_BYTE *)(a2 + 8);
  *(_QWORD *)(a1 + 40) = 0x800000000LL;
  *(_BYTE *)(a1 + 24) = v6;
  *(_QWORD *)(a1 + 32) = a1 + 48;
  if ( *(_DWORD *)(a2 + 24) )
  {
    sub_23FAD70(a1 + 32, a2 + 16, a1 + 48, a4, a5, a6);
    result = (unsigned int)(*(_DWORD *)(a1 + 8) + 1);
    *(_DWORD *)(a1 + 8) = result;
  }
  else
  {
    *(_DWORD *)(a1 + 8) = 1;
    return 1;
  }
  return result;
}
