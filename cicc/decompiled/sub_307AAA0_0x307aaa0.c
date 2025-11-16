// Function: sub_307AAA0
// Address: 0x307aaa0
//
char __fastcall sub_307AAA0(__int64 a1, __int64 a2)
{
  int v2; // eax

  if ( *(_DWORD *)(a2 + 1632) <= 0x1Du || *(_DWORD *)(a2 + 1624) <= 0x3Fu )
  {
    LOBYTE(v2) = 0;
  }
  else
  {
    if ( *(_BYTE *)a1 == 85 )
    {
      if ( (unsigned __int8)sub_A73ED0((_QWORD *)(a1 + 72), 36) || (unsigned __int8)sub_B49560(a1, 36) )
      {
        LOBYTE(v2) = *(_BYTE *)(**(_QWORD **)(*(_QWORD *)(a1 + 80) + 16LL) + 8LL) == 7;
        return v2;
      }
    }
    else if ( (unsigned __int8)sub_B2D610(a1, 36) && *(_BYTE *)(**(_QWORD **)(*(_QWORD *)(a1 + 24) + 16LL) + 8LL) == 7 )
    {
      return (unsigned int)sub_CE9220(a1) ^ 1;
    }
    LOBYTE(v2) = 0;
  }
  return v2;
}
