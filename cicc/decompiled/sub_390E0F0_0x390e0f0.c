// Function: sub_390E0F0
// Address: 0x390e0f0
//
__int64 __fastcall sub_390E0F0(__int64 a1, _QWORD *a2)
{
  __int64 v3; // rsi

  v3 = *(_QWORD *)(a1 + 8);
  if ( !v3 || v3 == *(_QWORD *)(a1 + 24) + 96LL )
    return sub_38D04A0(a2, *(_QWORD *)(a1 + 24));
  else
    return sub_38D01B0((__int64)a2, v3);
}
