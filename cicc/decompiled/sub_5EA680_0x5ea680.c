// Function: sub_5EA680
// Address: 0x5ea680
//
__int64 __fastcall sub_5EA680(__int64 a1, __int64 a2)
{
  __int64 v3; // rsi
  __int64 v4; // rsi
  __int64 result; // rax

  v3 = *(_QWORD *)(a1 + 456);
  if ( !v3 )
  {
    *(_QWORD *)(a1 + 456) = sub_5E4B20(0);
    sub_643E40(sub_5E9940, a1, 0);
    v3 = *(_QWORD *)(a1 + 456);
  }
  v4 = v3 + 128;
  if ( !a2 )
    return sub_6793E0(0, v4, 0, 0, 0);
  result = sub_6793E0(a2, v4, 0, 0, *(unsigned int *)(a2 + 36));
  *(_BYTE *)(a2 + 32) |= 4u;
  return result;
}
