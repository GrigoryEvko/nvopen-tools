// Function: sub_7281B0
// Address: 0x7281b0
//
void __fastcall sub_7281B0(__int64 a1, __int64 a2)
{
  char v2; // al

  if ( a1 )
  {
    v2 = *(_BYTE *)(a1 + 24);
    if ( v2 == 3 )
    {
      if ( (*(_BYTE *)(*(_QWORD *)(a1 + 56) - 8LL) & 1) != 0 )
        return;
      goto LABEL_6;
    }
    if ( v2 == 17 && (*(_BYTE *)(*(_QWORD *)(a1 + 56) - 8LL) & 1) == 0 )
    {
LABEL_6:
      *(_DWORD *)(a2 + 80) = 1;
      *(_DWORD *)(a2 + 72) = 1;
    }
  }
}
