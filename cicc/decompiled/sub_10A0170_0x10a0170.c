// Function: sub_10A0170
// Address: 0x10a0170
//
void __fastcall sub_10A0170(__int64 a1, __int64 a2)
{
  int v2; // eax

  *(_BYTE *)(a1 + 4) = 0;
  if ( a2 )
  {
    v2 = sub_B45210(a2);
    *(_BYTE *)(a1 + 4) = 1;
    *(_DWORD *)a1 = v2;
  }
}
