// Function: sub_F8D110
// Address: 0xf8d110
//
__int64 __fastcall sub_F8D110(__int64 a1, __int64 a2)
{
  __int64 v2; // r13
  __int64 v3; // rax

  v2 = sub_F894B0(a1, *(_QWORD *)(a2 + 32));
  v3 = sub_F7D780(a1, v2);
  return sub_F80B30(a1, v2, *(_QWORD *)(a2 + 40), 0x2Fu, v3);
}
