// Function: sub_B2BD80
// Address: 0xb2bd80
//
__int64 __fastcall sub_B2BD80(__int64 a1)
{
  __int64 result; // rax
  _QWORD v2[3]; // [rsp+8h] [rbp-18h] BYREF

  v2[0] = *(_QWORD *)(*(_QWORD *)(a1 + 24) + 120LL);
  result = sub_A74710(v2, *(_DWORD *)(a1 + 32) + 1, 51);
  if ( !(_BYTE)result )
    return sub_A74710(v2, *(_DWORD *)(a1 + 32) + 1, 50);
  return result;
}
