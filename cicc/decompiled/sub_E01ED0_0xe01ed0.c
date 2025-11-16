// Function: sub_E01ED0
// Address: 0xe01ed0
//
_BOOL8 __fastcall sub_E01ED0(unsigned __int8 *a1, __int64 a2, __int64 a3)
{
  __int64 v4; // rcx
  __int64 v5; // r8
  __int64 v6; // r9

  return !(unsigned __int8)sub_E00080(a1)
      || sub_E01E90((__int64)a1, *(_QWORD *)(a2 + 16), *(_QWORD *)(a3 + 16), v4, v5, v6);
}
