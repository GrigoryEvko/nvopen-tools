// Function: sub_6EE7B0
// Address: 0x6ee7b0
//
__int64 __fastcall sub_6EE7B0(__int64 a1)
{
  __int64 result; // rax
  char i; // dl
  __int64 v3; // rax

  result = **(_QWORD **)(a1 + 72);
  for ( i = *(_BYTE *)(result + 140); i == 12; i = *(_BYTE *)(result + 140) )
    result = *(_QWORD *)(result + 160);
  if ( i )
  {
    if ( (unsigned __int8)(*(_BYTE *)(a1 + 56) - 108) <= 1u )
    {
      result = sub_8D4870(result);
    }
    else if ( i == 6 )
    {
      result = sub_8D46C0(result);
    }
    else
    {
      if ( dword_4F04C44 == -1 )
      {
        v3 = qword_4F04C68[0] + 776LL * dword_4F04C64;
        if ( (*(_BYTE *)(v3 + 6) & 6) == 0 && *(_BYTE *)(v3 + 4) != 12 )
          sub_721090(a1);
      }
      result = *(_QWORD *)&dword_4D03B80;
    }
    while ( *(_BYTE *)(result + 140) == 12 )
      result = *(_QWORD *)(result + 160);
  }
  return result;
}
