// Function: sub_29C1BE0
// Address: 0x29c1be0
//
__int64 __fastcall sub_29C1BE0(__int64 a1, __int64 a2)
{
  int v2; // edx
  char v3; // bl
  __int64 v4; // rax
  __int64 v5; // rdx
  _QWORD v7[6]; // [rsp+0h] [rbp-30h] BYREF

  v2 = *(unsigned __int8 *)(a2 + 8);
  if ( (_BYTE)v2 != 12
    && (unsigned __int8)v2 > 3u
    && (_BYTE)v2 != 5
    && (v2 & 0xFB) != 0xA
    && (v2 & 0xFD) != 4
    && ((unsigned __int8)(v2 - 15) > 3u && v2 != 20 || !(unsigned __int8)sub_BCEBA0(a2, 0)) )
  {
    return 0;
  }
  v3 = sub_AE5020(a1 + 312, a2);
  v4 = sub_9208B0(a1 + 312, a2);
  v7[1] = v5;
  v7[0] = 8 * (((1LL << v3) + ((unsigned __int64)(v4 + 7) >> 3) - 1) >> v3 << v3);
  return (int)sub_CA1930(v7);
}
