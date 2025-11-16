// Function: sub_B192B0
// Address: 0xb192b0
//
__int64 __fastcall sub_B192B0(__int64 a1, __int64 a2)
{
  __int64 v2; // rdx
  unsigned int v3; // eax
  unsigned int v4; // r8d

  if ( a2 )
  {
    v2 = (unsigned int)(*(_DWORD *)(a2 + 44) + 1);
    v3 = *(_DWORD *)(a2 + 44) + 1;
  }
  else
  {
    v2 = 0;
    v3 = 0;
  }
  v4 = 0;
  if ( v3 < *(_DWORD *)(a1 + 32) )
    LOBYTE(v4) = *(_QWORD *)(*(_QWORD *)(a1 + 24) + 8 * v2) != 0;
  return v4;
}
