// Function: sub_318B4B0
// Address: 0x318b4b0
//
__int64 __fastcall sub_318B4B0(__int64 a1)
{
  __int64 v1; // rax
  __int64 v2; // rsi

  v1 = *(_QWORD *)(a1 + 16);
  v2 = *(_QWORD *)(v1 + 32);
  if ( v2 == *(_QWORD *)(v1 + 40) + 48LL || !v2 )
    return sub_3186770(*(_QWORD *)(a1 + 24), 0);
  else
    return sub_3186770(*(_QWORD *)(a1 + 24), v2 - 24);
}
