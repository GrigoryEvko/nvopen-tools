// Function: sub_D949F0
// Address: 0xd949f0
//
__int64 __fastcall sub_D949F0(__int64 a1, __int64 *a2, __int64 a3)
{
  unsigned int v4; // edx
  unsigned __int64 v5; // rax
  int v6; // eax

  v4 = *(_DWORD *)(a3 + 8);
  if ( v4 > 0x40 )
  {
    sub_C43D10(a3);
  }
  else
  {
    v5 = (0xFFFFFFFFFFFFFFFFLL >> -(char)v4) & ~*(_QWORD *)a3;
    if ( !v4 )
      v5 = 0;
    *(_QWORD *)a3 = v5;
  }
  sub_C46250(a3);
  sub_C45EE0(a3, a2);
  v6 = *(_DWORD *)(a3 + 8);
  *(_DWORD *)(a3 + 8) = 0;
  *(_DWORD *)(a1 + 8) = v6;
  *(_QWORD *)a1 = *(_QWORD *)a3;
  return a1;
}
