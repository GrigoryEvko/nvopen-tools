// Function: sub_3713950
// Address: 0x3713950
//
__int64 *__fastcall sub_3713950(__int64 *a1, __int64 a2, __int64 a3, __int64 a4)
{
  char v4; // al
  _BYTE v6[9]; // [rsp+Fh] [rbp-11h] BYREF

  v4 = *(_BYTE *)(a2 + 10);
  if ( v4 )
    v4 = *(_WORD *)(a2 + 8) == 4614;
  v6[0] = v4;
  sub_3713670(a1, v6, (_QWORD *)(a2 + 16), a4);
  return a1;
}
