// Function: sub_D036D0
// Address: 0xd036d0
//
void __fastcall sub_D036D0(__int64 a1, char a2)
{
  int v2; // eax
  int v3; // eax

  if ( a2 && (*(_BYTE *)(a1 + 1) & 1) != 0 )
  {
    v2 = *(int *)a1 >> 9;
    if ( v2 != -4194304 )
    {
      v3 = -512 * v2;
      BYTE1(v3) |= 1u;
      *(_DWORD *)a1 = (unsigned __int8)*(_DWORD *)a1 | v3;
    }
  }
}
