// Function: sub_22F4E30
// Address: 0x22f4e30
//
__int64 __fastcall sub_22F4E30(_DWORD *a1, __int64 a2, __int64 a3, __int64 a4, unsigned int a5)
{
  int v5; // eax

  v5 = *(_DWORD *)(a2 + 48);
  if ( !*a1 || (a5 = 1, (v5 & *a1) != 0) )
    LOBYTE(a5) = (a1[1] & v5) != 0;
  return a5;
}
