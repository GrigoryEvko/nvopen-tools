// Function: sub_2C0D6E0
// Address: 0x2c0d6e0
//
__int64 __fastcall sub_2C0D6E0(__int64 a1, __int64 a2)
{
  _QWORD *v2; // rax
  _QWORD *v4; // rsi
  unsigned int v5; // r8d
  __int64 v6; // [rsp+8h] [rbp-8h] BYREF

  v2 = *(_QWORD **)(a1 + 48);
  v6 = a2;
  if ( *v2 != a2 )
    return 0;
  v4 = &v2[*(_DWORD *)(a1 + 56) - (1 - ((unsigned int)(*(_BYTE *)(a1 + 104) == 0) - 1)) + 1];
  LOBYTE(v5) = v4 == sub_2C0D620(v2 + 1, (__int64)v4, &v6);
  return v5;
}
