// Function: sub_877E20
// Address: 0x877e20
//
void __fastcall sub_877E20(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  __int64 v6; // rax

  if ( a3 )
  {
    if ( *(_BYTE *)(a3 + 140) == 14 )
      a3 = sub_7CFE40(a3, a2, a3, a4, a5, a6);
    if ( a1 )
    {
      *(_BYTE *)(a1 + 81) |= 0x10u;
      *(_QWORD *)(a1 + 64) = a3;
    }
    if ( a2 )
    {
      v6 = *(_QWORD *)(*(_QWORD *)(a3 + 168) + 152LL);
      if ( v6 )
      {
        *(_BYTE *)(a2 + 89) |= 4u;
        *(_QWORD *)(a2 + 40) = v6;
      }
    }
  }
}
