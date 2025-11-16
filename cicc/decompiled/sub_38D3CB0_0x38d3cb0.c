// Function: sub_38D3CB0
// Address: 0x38d3cb0
//
__int64 __fastcall sub_38D3CB0(__int64 *a1, __int64 a2)
{
  __int64 v2; // rax

  v2 = sub_38BFA60(a1[1], 1);
  *(_QWORD *)(a2 + 8) = v2;
  return (*(__int64 (__fastcall **)(__int64 *, __int64, _QWORD))(*a1 + 176))(a1, v2, 0);
}
