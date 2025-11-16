// Function: sub_1487E90
// Address: 0x1487e90
//
__int64 __fastcall sub_1487E90(__int64 a1, __int64 a2, __int64 a3)
{
  __int64 v3; // r13
  __int64 v4; // rax
  __int64 v6; // r14

  if ( (*(_BYTE *)(a3 + 18) & 1) == 0 )
  {
    v3 = *(_QWORD *)(a3 - 24);
    if ( *(_BYTE *)(v3 + 16) == 56 )
    {
      v6 = *(_QWORD *)(v3 - 24LL * (*(_DWORD *)(v3 + 20) & 0xFFFFFFF));
      if ( *(_BYTE *)(v6 + 16) == 3
        && (*(_BYTE *)(v6 + 80) & 1) != 0
        && !(unsigned __int8)sub_15E4F60(*(_QWORD *)(v3 - 24LL * (*(_DWORD *)(v3 + 20) & 0xFFFFFFF))) )
      {
        __asm { jmp     rax }
      }
    }
  }
  v4 = sub_1456E90(a2);
  sub_14573F0(a1, v4);
  return a1;
}
