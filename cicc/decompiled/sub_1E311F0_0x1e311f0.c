// Function: sub_1E311F0
// Address: 0x1e311f0
//
void __fastcall sub_1E311F0(__int64 a1, __int64 a2, __int64 a3)
{
  int v3; // eax

  if ( ((*(_DWORD *)a1 >> 8) & 0xFFF) != 0
    && (LODWORD(a2) = sub_38D6F10(a3 + 8, a2, (*(_DWORD *)a1 >> 8) & 0xFFF),
        v3 = *(_DWORD *)a1,
        *(_DWORD *)a1 &= 0xFFF000FF,
        (v3 & 0x10000000) != 0) )
  {
    *(_BYTE *)(a1 + 4) &= ~1u;
    sub_1E310D0(a1, a2);
  }
  else
  {
    sub_1E310D0(a1, a2);
  }
}
