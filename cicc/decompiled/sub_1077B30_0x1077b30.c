// Function: sub_1077B30
// Address: 0x1077b30
//
__int64 __fastcall sub_1077B30(__int64 a1, __int64 *a2)
{
  _QWORD *v3; // rbx
  __int64 result; // rax
  __int64 v5; // rsi
  __int64 v6; // rsi

  v3 = **(_QWORD ***)(a1 + 104);
  result = (*(__int64 (__fastcall **)(_QWORD *))(*v3 + 80LL))(v3);
  v5 = result + v3[4] - v3[2];
  if ( v5 )
  {
    v6 = v5 - a2[1];
    if ( v6 != (unsigned int)v6 )
      sub_C64ED0("section size does not fit in a uint32_t", 1u);
    return sub_1076D00(**(_QWORD **)(a1 + 104), v6, *a2);
  }
  return result;
}
