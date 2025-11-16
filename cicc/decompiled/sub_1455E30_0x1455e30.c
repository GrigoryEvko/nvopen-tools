// Function: sub_1455E30
// Address: 0x1455e30
//
__int64 __fastcall sub_1455E30(__int64 a1, __int64 a2, __int64 a3)
{
  unsigned int v4; // ecx
  int v5; // eax

  v4 = *(_DWORD *)(a3 + 8);
  if ( v4 > 0x40 )
    sub_16A8F40(a3);
  else
    *(_QWORD *)a3 = ~*(_QWORD *)a3 & (0xFFFFFFFFFFFFFFFFLL >> -(char)v4);
  sub_16A7400(a3);
  sub_16A7200(a3, a2);
  v5 = *(_DWORD *)(a3 + 8);
  *(_DWORD *)(a3 + 8) = 0;
  *(_DWORD *)(a1 + 8) = v5;
  *(_QWORD *)a1 = *(_QWORD *)a3;
  return a1;
}
