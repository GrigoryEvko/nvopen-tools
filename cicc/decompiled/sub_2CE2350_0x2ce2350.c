// Function: sub_2CE2350
// Address: 0x2ce2350
//
void __fastcall sub_2CE2350(__int64 a1, __int64 a2)
{
  __int64 v3; // rsi
  _BYTE *v4; // rsi
  __int64 v5[3]; // [rsp+8h] [rbp-18h] BYREF

  v3 = *(_QWORD *)(a2 - 64);
  if ( *(_BYTE *)(*(_QWORD *)(v3 + 8) + 8LL) == 15 )
  {
    sub_2CE18D0(*(_QWORD *)(a2 - 32), v3, *(_BYTE *)(a2 + 2) & 1, a2);
    v5[0] = a2;
    v4 = *(_BYTE **)(a1 + 432);
    if ( v4 == *(_BYTE **)(a1 + 440) )
    {
      sub_249A840(a1 + 424, v4, v5);
    }
    else
    {
      if ( v4 )
      {
        *(_QWORD *)v4 = a2;
        v4 = *(_BYTE **)(a1 + 432);
      }
      *(_QWORD *)(a1 + 432) = v4 + 8;
    }
  }
}
