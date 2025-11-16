// Function: sub_2EAB1E0
// Address: 0x2eab1e0
//
void __fastcall sub_2EAB1E0(__int64 a1, unsigned int a2, _QWORD *a3)
{
  int v3; // eax

  if ( ((*(_DWORD *)a1 >> 8) & 0xFFF) != 0
    && (a2 = sub_E91CF0(a3, a2, (*(_DWORD *)a1 >> 8) & 0xFFF),
        v3 = *(_DWORD *)a1,
        *(_DWORD *)a1 &= 0xFFF000FF,
        (v3 & 0x10000000) != 0) )
  {
    *(_BYTE *)(a1 + 4) &= ~1u;
    sub_2EAB0C0(a1, a2);
  }
  else
  {
    sub_2EAB0C0(a1, a2);
  }
}
