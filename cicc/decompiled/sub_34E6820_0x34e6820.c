// Function: sub_34E6820
// Address: 0x34e6820
//
bool __fastcall sub_34E6820(__int64 a1, __int64 a2)
{
  __int64 v2; // r12
  __int64 v3; // rcx
  __int64 v4; // r13
  __int64 v6; // rsi

  v2 = *(_QWORD *)(a1 + 8);
  v3 = *(_QWORD *)(a1 + 32);
  if ( a2 == v2 )
  {
    v6 = *(_QWORD *)(a1 + 8);
    v2 = a1;
    return sub_2E322C0(v2, v6);
  }
  else
  {
    v4 = v3 + 320;
    if ( v3 + 320 != v2 )
    {
      while ( v2 + 48 == (*(_QWORD *)(v2 + 48) & 0xFFFFFFFFFFFFFFF8LL) && sub_2E322C0(a1, v2) )
      {
        v6 = *(_QWORD *)(v2 + 8);
        if ( a2 == v6 )
          return sub_2E322C0(v2, v6);
        a1 = v2;
        if ( v4 == v6 )
          break;
        v2 = *(_QWORD *)(v2 + 8);
      }
    }
    return 0;
  }
}
