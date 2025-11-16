// Function: sub_12A71A0
// Address: 0x12a71a0
//
__int64 __fastcall sub_12A71A0(__int64 a1)
{
  __int64 result; // rax
  _DWORD v2[3]; // [rsp+1Ch] [rbp-14h] BYREF

  result = sub_620FD0(*(_QWORD *)(a1 + 56), v2);
  if ( v2[0] )
    sub_127B550("unexpected constant overflow in __wgmma_mma_async operand", (_DWORD *)(a1 + 36), 1);
  return result;
}
