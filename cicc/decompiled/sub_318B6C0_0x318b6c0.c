// Function: sub_318B6C0
// Address: 0x318b6c0
//
__int64 __fastcall sub_318B6C0(__int64 a1)
{
  __int64 v1; // rdx
  __int64 v2; // rsi

  v1 = *(_QWORD *)(a1 + 16);
  if ( (*(_DWORD *)(v1 + 4) & 0x7FFFFFF) != 0 && (v2 = *(_QWORD *)(v1 - 32LL * (*(_DWORD *)(v1 + 4) & 0x7FFFFFF))) != 0 )
    return sub_3186770(*(_QWORD *)(a1 + 24), v2);
  else
    return 0;
}
