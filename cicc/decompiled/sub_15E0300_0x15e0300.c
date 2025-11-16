// Function: sub_15E0300
// Address: 0x15e0300
//
__int64 __fastcall sub_15E0300(__int64 a1)
{
  __int64 result; // rax
  int v2; // esi
  _QWORD v3[4]; // [rsp-20h] [rbp-20h] BYREF

  result = 0;
  if ( *(_BYTE *)(*(_QWORD *)a1 + 8LL) == 15 )
  {
    v2 = *(_DWORD *)(a1 + 32);
    v3[0] = *(_QWORD *)(*(_QWORD *)(a1 + 24) + 112LL);
    result = sub_1560290(v3, v2, 6);
    if ( !(_BYTE)result )
      return sub_1560290(v3, *(_DWORD *)(a1 + 32), 11);
  }
  return result;
}
