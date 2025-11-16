// Function: sub_8E38C0
// Address: 0x8e38c0
//
_BOOL8 __fastcall sub_8E38C0(__int64 a1, __int64 a2)
{
  return sub_8D2930(a1) && sub_8D2930(a2) && *(_QWORD *)(a1 + 128) == *(_QWORD *)(a2 + 128);
}
