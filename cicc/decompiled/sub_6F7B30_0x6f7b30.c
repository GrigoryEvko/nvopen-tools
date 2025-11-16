// Function: sub_6F7B30
// Address: 0x6f7b30
//
__int64 __fastcall sub_6F7B30(__int64 a1, unsigned __int8 a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  __int64 v8; // rax
  __int64 *v9; // rax
  __int64 result; // rax

  v8 = sub_6F6F40((const __m128i *)a1, 0, a3, a4, a5, a6);
  v9 = (__int64 *)sub_73DBF0(a2, a3, v8);
  sub_6E70E0(v9, a4);
  *(_QWORD *)(a4 + 68) = *(_QWORD *)(a1 + 68);
  result = *(_QWORD *)(a1 + 76);
  *(_QWORD *)(a4 + 76) = result;
  return result;
}
