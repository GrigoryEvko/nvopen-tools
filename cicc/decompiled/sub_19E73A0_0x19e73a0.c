// Function: sub_19E73A0
// Address: 0x19e73a0
//
__int64 __fastcall sub_19E73A0(__int64 a1, __int64 a2)
{
  __int64 v2; // r12
  __int64 v4; // [rsp+0h] [rbp-20h] BYREF
  __int64 v5; // [rsp+8h] [rbp-18h] BYREF

  if ( *(_BYTE *)(a2 + 16) <= 0x17u )
  {
    if ( *(_BYTE *)(a2 + 16) != 23 )
      BUG();
    return *(_QWORD *)(a2 + 64);
  }
  else
  {
    v2 = *(_QWORD *)(a2 + 40);
    if ( !v2 )
    {
      v4 = a2;
      if ( (unsigned __int8)sub_19E72F0(a1 + 1672, &v4, &v5) )
        return *(_QWORD *)(v5 + 8);
    }
    return v2;
  }
}
