// Function: sub_1E36CE0
// Address: 0x1e36ce0
//
__int64 __fastcall sub_1E36CE0(__int64 a1, __int64 a2, __int64 a3)
{
  __int64 v3; // rsi

  v3 = *(_QWORD *)(a2 + 8);
  if ( v3 )
    sub_1DDC460(a1, v3, a3);
  else
    *(_BYTE *)(a1 + 8) = 0;
  return a1;
}
