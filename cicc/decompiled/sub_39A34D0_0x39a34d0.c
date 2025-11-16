// Function: sub_39A34D0
// Address: 0x39a34d0
//
__int64 __fastcall sub_39A34D0(__int64 a1, __int64 a2, __int16 a3)
{
  unsigned __int16 v4; // ax
  __int64 v6[6]; // [rsp+0h] [rbp-30h] BYREF

  v4 = sub_398C0A0(*(_QWORD *)(a1 + 200));
  LODWORD(v6[0]) = 1;
  WORD2(v6[0]) = a3;
  v6[1] = 1;
  if ( v4 <= 3u )
    HIWORD(v6[0]) = 12;
  else
    HIWORD(v6[0]) = 25;
  return sub_39A31C0((__int64 *)(a2 + 8), (__int64 *)(a1 + 88), v6);
}
