// Function: sub_310C730
// Address: 0x310c730
//
__int64 __fastcall sub_310C730(__int64 a1, _QWORD *a2, unsigned int *a3)
{
  _BYTE *v3; // rax
  unsigned int v4; // r8d
  _BYTE *v6; // rax

  v3 = (_BYTE *)*a2;
  v4 = 0;
  if ( *(_BYTE *)*a2 != 85 )
    return 0;
  if ( (v3[7] & 0x40) != 0 )
    v6 = (_BYTE *)*((_QWORD *)v3 - 1);
  else
    v6 = &v3[-32 * (*((_DWORD *)v3 + 1) & 0x7FFFFFF)];
  LOBYTE(v4) = **(_BYTE **)&v6[32 * *a3] <= 0x15u;
  return v4;
}
