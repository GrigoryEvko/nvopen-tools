// Function: sub_826940
// Address: 0x826940
//
__int64 __fastcall sub_826940(_DWORD *a1, _DWORD *a2)
{
  __int64 result; // rax
  __int64 v5; // rcx
  __int64 v6; // rdx
  char v7; // si

  *a2 = 0;
  *a1 = 0;
  result = *(_QWORD *)(qword_4F04C50 + 32LL);
  if ( result )
  {
    while ( (*(_BYTE *)(result + 206) & 2) != 0 )
    {
      v5 = *(_QWORD *)(*(_QWORD *)(result + 40) + 32LL);
      v6 = *(_QWORD *)(v5 + 168);
      v7 = *(_BYTE *)(v6 + 109);
      if ( (v7 & 0x40) != 0 || (*(_BYTE *)(v6 + 92) & 4) != 0 )
      {
        *a1 = (v7 & 0x40) != 0;
        *a2 = (*(_BYTE *)(v6 + 92) & 4) != 0;
        return result;
      }
      result = *(_QWORD *)(v5 + 48);
      if ( !result )
        return result;
    }
  }
  return result;
}
