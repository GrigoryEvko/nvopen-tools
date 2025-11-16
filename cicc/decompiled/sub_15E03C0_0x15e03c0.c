// Function: sub_15E03C0
// Address: 0x15e03c0
//
__int64 __fastcall sub_15E03C0(__int64 a1)
{
  int v1; // esi
  __int64 result; // rax
  _QWORD v3[3]; // [rsp+8h] [rbp-18h] BYREF

  v1 = *(_DWORD *)(a1 + 32);
  v3[0] = *(_QWORD *)(*(_QWORD *)(a1 + 24) + 112LL);
  result = sub_1560290(v3, v1, 37);
  if ( !(_BYTE)result )
    return sub_1560290(v3, *(_DWORD *)(a1 + 32), 36);
  return result;
}
