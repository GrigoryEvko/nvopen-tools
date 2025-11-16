// Function: sub_6E44B0
// Address: 0x6e44b0
//
__int64 __fastcall sub_6E44B0(__int64 a1, __int64 *a2, _QWORD *a3)
{
  __int64 v4; // r15
  __int64 v5; // rbx
  __int64 result; // rax

  v4 = *(_QWORD *)(a1 + 16);
  v5 = sub_6E3DA0(v4, 0);
  *a2 = sub_6E3F00(*(_QWORD *)(v4 + 56), a1, v5);
  result = *(_QWORD *)(v5 + 68);
  *a3 = result;
  return result;
}
