// Function: sub_AB0A00
// Address: 0xab0a00
//
__int64 __fastcall sub_AB0A00(__int64 a1, __int64 a2)
{
  unsigned int v2; // eax
  unsigned int v4; // eax

  if ( sub_AAF760(a2) || sub_AAFBB0(a2) )
  {
    v2 = *(_DWORD *)(a2 + 8);
    *(_DWORD *)(a1 + 8) = v2;
    if ( v2 <= 0x40 )
    {
      *(_QWORD *)a1 = 0;
      return a1;
    }
    sub_C43690(a1, 0, 0);
    return a1;
  }
  else
  {
    v4 = *(_DWORD *)(a2 + 8);
    *(_DWORD *)(a1 + 8) = v4;
    if ( v4 > 0x40 )
    {
      sub_C43780(a1, a2);
      return a1;
    }
    *(_QWORD *)a1 = *(_QWORD *)a2;
    return a1;
  }
}
