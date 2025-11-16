// Function: sub_39A3990
// Address: 0x39a3990
//
__int64 __fastcall sub_39A3990(__int64 a1, __int64 *a2, __int16 a3, __int16 a4, __int64 a5)
{
  __int64 v6[2]; // [rsp+0h] [rbp-10h] BYREF

  WORD2(v6[0]) = a3;
  LODWORD(v6[0]) = 4;
  HIWORD(v6[0]) = a4;
  v6[1] = a5;
  return sub_39A31C0(a2, (__int64 *)(a1 + 88), v6);
}
