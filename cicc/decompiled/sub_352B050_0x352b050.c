// Function: sub_352B050
// Address: 0x352b050
//
char __fastcall sub_352B050(__int64 a1)
{
  int v1; // edx
  __int64 v2; // rax

  if ( (unsigned int)*(unsigned __int16 *)(a1 + 68) - 1 <= 1 && (*(_BYTE *)(*(_QWORD *)(a1 + 32) + 64LL) & 0x20) != 0 )
  {
    LOBYTE(v2) = 1;
  }
  else
  {
    v1 = *(_DWORD *)(a1 + 44);
    LOBYTE(v2) = 0;
    if ( (v1 & 0x20000) == 0 )
    {
      if ( (v1 & 4) != 0 || (v1 & 8) == 0 )
        return (*(_QWORD *)(*(_QWORD *)(a1 + 16) + 24LL) >> 36) & 1LL;
      else
        LOBYTE(v2) = sub_2E88A90(a1, 0x1000000000LL, 1);
    }
  }
  return v2;
}
