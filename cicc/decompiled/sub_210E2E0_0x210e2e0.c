// Function: sub_210E2E0
// Address: 0x210e2e0
//
__int64 __fastcall sub_210E2E0(__int64 a1, __int64 a2, bool *a3)
{
  __int64 result; // rax
  __int64 v4; // rcx
  int v5; // ecx

  result = 0;
  if ( *(_BYTE *)(a2 + 16) == 78 )
  {
    v4 = *(_QWORD *)(a2 - 24);
    if ( !*(_BYTE *)(v4 + 16) && (*(_BYTE *)(v4 + 33) & 0x20) != 0 )
    {
      v5 = *(_DWORD *)(v4 + 36);
      if ( (unsigned int)(v5 - 116) <= 1 )
      {
        *a3 = v5 == 117;
        return 1;
      }
    }
  }
  return result;
}
