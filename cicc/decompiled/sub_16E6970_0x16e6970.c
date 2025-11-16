// Function: sub_16E6970
// Address: 0x16e6970
//
__int64 __fastcall sub_16E6970(__int64 a1, _BYTE *a2)
{
  __int64 *v3; // r9
  _QWORD *v4; // rsi

  v3 = *(__int64 **)(a1 + 224);
  v4 = *(_QWORD **)(a1 + 264);
  *(_DWORD *)(a1 + 248) = 0;
  *(_QWORD *)(a1 + 240) = v3;
  if ( *(_DWORD *)(v4[1] + 32LL) == 5 )
    sub_16E62D0(a1 + 224, v3, 0, (__int64)(v4[3] - v4[2]) >> 3, 0);
  else
    sub_16E42A0(a1, (__int64)v4);
  *a2 = 1;
  return 1;
}
