// Function: sub_307AA70
// Address: 0x307aa70
//
__int64 __fastcall sub_307AA70(__int64 a1)
{
  unsigned int v1; // r8d
  __int64 v2; // rax

  v1 = 0;
  if ( (unsigned int)*(unsigned __int16 *)(a1 + 68) - 2676 > 1 )
    return 0;
  v2 = *(_QWORD *)(a1 + 32);
  if ( *(_BYTE *)(v2 + 120) != 1 )
    return 0;
  LOBYTE(v1) = *(_QWORD *)(v2 + 144) == 101;
  return v1;
}
