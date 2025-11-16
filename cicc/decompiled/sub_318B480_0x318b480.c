// Function: sub_318B480
// Address: 0x318b480
//
__int64 __fastcall sub_318B480(__int64 a1, __int64 a2)
{
  __int64 v2; // rdx
  __int64 v3; // rcx
  __int64 v5; // rsi

  v2 = *(_QWORD *)(a2 + 16);
  v3 = *(_QWORD *)(a2 + 24);
  v5 = *(_QWORD *)(v2 + 40);
  *(_QWORD *)(a1 + 24) = v3;
  *(_QWORD *)(a1 + 8) = v2 + 24;
  *(_QWORD *)a1 = v5;
  *(_WORD *)(a1 + 16) = 0;
  return a1;
}
