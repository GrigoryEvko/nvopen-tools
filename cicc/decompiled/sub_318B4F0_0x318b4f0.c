// Function: sub_318B4F0
// Address: 0x318b4f0
//
__int64 __fastcall sub_318B4F0(__int64 a1)
{
  __int64 v1; // rsi

  v1 = *(_QWORD *)(*(_QWORD *)(a1 + 16) + 40LL);
  if ( v1 )
    return sub_3186770(*(_QWORD *)(a1 + 24), v1);
  else
    return 0;
}
