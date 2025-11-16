// Function: sub_802E80
// Address: 0x802e80
//
__int64 __fastcall sub_802E80(__int64 a1, __int64 a2, __int64 a3, __m128i *a4)
{
  __int64 result; // rax
  __int64 v7; // rax
  int v8[9]; // [rsp+Ch] [rbp-24h] BYREF

  result = *(_QWORD *)(a1 + 168) & 0xFF0000020000LL;
  if ( result == 0xA0000020000LL )
  {
    v8[0] = 0;
    if ( (*(_BYTE *)(a1 - 8) & 1) != 0 )
      sub_7296C0(v8);
    sub_801E60(a1, 0, a4);
    if ( a2 && *(char *)(a2 + 50) < 0 )
    {
      if ( !a3 )
      {
        v7 = *(_QWORD *)(a2 + 8);
        if ( v7 )
          a3 = *(_QWORD *)(v7 + 120);
        else
          a3 = *(_QWORD *)(a1 + 128);
      }
      *(_BYTE *)(a2 + 50) = ((unsigned __int8)sub_7F7570(a1, a3) << 7) | *(_BYTE *)(a2 + 50) & 0x7F;
    }
    result = (__int64)sub_729730(v8[0]);
    *(_BYTE *)(a1 + 170) &= ~2u;
  }
  return result;
}
