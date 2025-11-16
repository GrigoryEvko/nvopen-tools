// Function: sub_13A3810
// Address: 0x13a3810
//
__int64 __fastcall sub_13A3810(__int64 a1, __int64 a2, __int64 a3)
{
  __int64 v3; // r12
  unsigned int v5; // eax

  v3 = a2;
  if ( (int)sub_16AEA10(a2, a3) >= 0 )
    v3 = a3;
  v5 = *(_DWORD *)(v3 + 8);
  *(_DWORD *)(a1 + 8) = v5;
  if ( v5 > 0x40 )
    sub_16A4FD0(a1, v3);
  else
    *(_QWORD *)a1 = *(_QWORD *)v3;
  return a1;
}
