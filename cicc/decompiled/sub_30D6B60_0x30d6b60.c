// Function: sub_30D6B60
// Address: 0x30d6b60
//
__int64 __fastcall sub_30D6B60(__int64 a1, unsigned int a2, int a3)
{
  int v3; // esi
  int v5; // eax

  if ( a2 > 2 )
  {
    sub_30D6950(a1, 250);
    if ( *(_BYTE *)(a1 + 48) )
    {
      *(_DWORD *)(a1 + 44) = dword_5030408;
    }
    else
    {
      v5 = dword_5030408;
      *(_BYTE *)(a1 + 48) = 1;
      *(_DWORD *)(a1 + 44) = v5;
    }
    return a1;
  }
  else
  {
    if ( a3 == 1 )
    {
      v3 = 50;
    }
    else
    {
      v3 = 5;
      if ( a3 != 2 )
        v3 = qword_5030E88;
    }
    sub_30D6950(a1, v3);
    return a1;
  }
}
