// Function: sub_1455DC0
// Address: 0x1455dc0
//
__int64 __fastcall sub_1455DC0(__int64 a1, __int64 a2)
{
  unsigned int v2; // ecx
  int v3; // eax

  v2 = *(_DWORD *)(a2 + 8);
  if ( v2 > 0x40 )
    sub_16A8F40(a2);
  else
    *(_QWORD *)a2 = ~*(_QWORD *)a2 & (0xFFFFFFFFFFFFFFFFLL >> -(char)v2);
  sub_16A7400(a2);
  v3 = *(_DWORD *)(a2 + 8);
  *(_DWORD *)(a2 + 8) = 0;
  *(_DWORD *)(a1 + 8) = v3;
  *(_QWORD *)a1 = *(_QWORD *)a2;
  return a1;
}
