// Function: sub_2EAFC50
// Address: 0x2eafc50
//
_BYTE *__fastcall sub_2EAFC50(__int64 **a1, __int64 a2)
{
  __int64 v2; // rax
  unsigned __int64 v3; // rbx
  __int64 v4; // r13
  _BYTE *result; // rax

  sub_2EAFC20((__int64)a1, (_QWORD *)a2);
  v2 = sub_B2BE50(**a1);
  v3 = *(_QWORD *)(a2 + 64);
  v4 = v2;
  if ( !*(_BYTE *)(a2 + 72) )
    v3 = 0;
  result = (_BYTE *)sub_B6E940(v2);
  if ( (unsigned __int64)result <= v3 )
    return sub_B6EB20(v4, a2);
  return result;
}
