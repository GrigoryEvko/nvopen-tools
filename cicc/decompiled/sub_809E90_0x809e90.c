// Function: sub_809E90
// Address: 0x809e90
//
__int64 __fastcall sub_809E90(__int64 a1)
{
  char v1; // dl
  _DWORD v3[4]; // [rsp+0h] [rbp-10h] BYREF

  v1 = (_BYTE)dword_4F18B88 << 7;
  *(_BYTE *)(a1 + 89) = ((_BYTE)dword_4F18B88 << 7) | *(_BYTE *)(a1 + 89) & 0x7F;
  if ( (unsigned __int8)(*(_BYTE *)(a1 + 140) - 9) <= 2u && (*(_BYTE *)(a1 + 177) & 0x10) != 0 )
    *(_BYTE *)(*(_QWORD *)(*(_QWORD *)(a1 + 168) + 160LL) + 89LL) = *(_BYTE *)(*(_QWORD *)(*(_QWORD *)(a1 + 168) + 160LL)
                                                                             + 89LL)
                                                                  & 0x7F
                                                                  | v1;
  sub_737670(a1, 6u, (__int64 (__fastcall *)(__int64, _QWORD, _DWORD *))sub_80AE00, v3, 7);
  return 0;
}
