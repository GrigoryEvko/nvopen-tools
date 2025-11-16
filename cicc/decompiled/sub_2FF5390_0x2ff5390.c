// Function: sub_2FF5390
// Address: 0x2ff5390
//
__int64 __fastcall sub_2FF5390(__int64 a1, int a2, __int64 a3)
{
  unsigned __int64 v5; // rax
  __int16 v6; // dx

  do
  {
    v5 = sub_2EBEE10(a3, a2);
    v6 = *(_WORD *)(v5 + 68);
    if ( v6 == 20 )
    {
      a2 = *(_DWORD *)(*(_QWORD *)(v5 + 32) + 48LL);
    }
    else
    {
      if ( v6 != 12 )
        return (unsigned int)a2;
      a2 = *(_DWORD *)(*(_QWORD *)(v5 + 32) + 88LL);
    }
  }
  while ( a2 < 0 );
  return (unsigned int)a2;
}
