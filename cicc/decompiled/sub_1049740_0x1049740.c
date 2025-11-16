// Function: sub_1049740
// Address: 0x1049740
//
_BYTE *__fastcall sub_1049740(__int64 *a1, __int64 a2)
{
  unsigned __int64 v3; // r13
  __int64 v4; // rdi
  __int64 v5; // rax
  _BYTE *result; // rax
  __int64 v7; // rax

  sub_1049710((__int64)a1, (_QWORD *)a2);
  v3 = *(_QWORD *)(a2 + 64);
  v4 = *a1;
  if ( !*(_BYTE *)(a2 + 72) )
    v3 = 0;
  v5 = sub_B2BE50(v4);
  result = (_BYTE *)sub_B6E940(v5);
  if ( (unsigned __int64)result <= v3 )
  {
    v7 = sub_B2BE50(*a1);
    return sub_B6EB20(v7, a2);
  }
  return result;
}
