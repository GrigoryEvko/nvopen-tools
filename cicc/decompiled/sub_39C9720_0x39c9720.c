// Function: sub_39C9720
// Address: 0x39c9720
//
__int64 __fastcall sub_39C9720(__int64 a1, __int64 a2, __int16 a3, unsigned int a4)
{
  unsigned __int16 v6; // ax
  __int64 v8[6]; // [rsp+0h] [rbp-30h] BYREF

  v6 = sub_398C0A0(*(_QWORD *)(a1 + 200));
  WORD2(v8[0]) = a3;
  LODWORD(v8[0]) = 9;
  HIWORD(v8[0]) = v6 < 4u ? 6 : 23;
  v8[1] = a4;
  return sub_39A31C0((__int64 *)(a2 + 8), (__int64 *)(a1 + 88), v8);
}
