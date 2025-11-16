// Function: sub_137A320
// Address: 0x137a320
//
__int64 __fastcall sub_137A320(__int64 a1, __int64 a2)
{
  __int64 v2; // rax
  unsigned int v3; // r8d
  __int64 v5; // rax
  int v6; // edx
  bool v7; // r15
  _BOOL4 v8; // r14d
  int v9[9]; // [rsp+Ch] [rbp-24h] BYREF

  v2 = sub_157EBA0(a2);
  v3 = 0;
  if ( *(_BYTE *)(v2 + 16) == 26 && (*(_DWORD *)(v2 + 20) & 0xFFFFFFF) == 3 )
  {
    v5 = *(_QWORD *)(v2 - 72);
    if ( *(_BYTE *)(v5 + 16) == 75 )
    {
      v6 = *(unsigned __int16 *)(v5 + 18);
      BYTE1(v6) &= ~0x80u;
      if ( (unsigned int)(v6 - 32) <= 1 && *(_BYTE *)(**(_QWORD **)(v5 - 48) + 8LL) == 15 )
      {
        v7 = v6 != 33;
        v8 = v6 == 33;
        sub_16AF710(v9, 20, 32);
        sub_1379150(a1, a2, v7, v9[0]);
        sub_1379150(a1, a2, v8, 0x80000000 - v9[0]);
        return 1;
      }
    }
  }
  return v3;
}
