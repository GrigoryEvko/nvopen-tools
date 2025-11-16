// Function: sub_137A790
// Address: 0x137a790
//
__int64 __fastcall sub_137A790(__int64 a1, __int64 a2)
{
  __int64 v2; // r8
  __int64 result; // rax
  int v4[5]; // [rsp+Ch] [rbp-14h] BYREF

  v2 = sub_157EBA0(a2);
  result = 0;
  if ( *(_BYTE *)(v2 + 16) == 29 )
  {
    sub_16AF710(v4, 0xFFFFF, 0x100000);
    sub_1379150(a1, a2, 0, v4[0]);
    sub_1379150(a1, a2, 1, 0x80000000 - v4[0]);
    return 1;
  }
  return result;
}
