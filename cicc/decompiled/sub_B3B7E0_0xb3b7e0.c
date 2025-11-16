// Function: sub_B3B7E0
// Address: 0xb3b7e0
//
__int64 __fastcall sub_B3B7E0(__int64 a1, __int64 a2)
{
  __int64 result; // rax
  _QWORD v3[2]; // [rsp+0h] [rbp-10h] BYREF

  v3[0] = *(_QWORD *)(a1 + 24);
  result = *(_QWORD *)(a1 + 32);
  *(_DWORD *)(a2 + 8) = 0;
  v3[1] = result;
  if ( result )
    return sub_C937F0(v3, a2, "\n\t", 2, 0xFFFFFFFFLL, 0);
  return result;
}
