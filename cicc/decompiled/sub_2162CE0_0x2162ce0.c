// Function: sub_2162CE0
// Address: 0x2162ce0
//
_BYTE *__fastcall sub_2162CE0(__int64 a1, __int64 a2, __int64 a3)
{
  _QWORD *v4; // rdx
  _BYTE *result; // rax

  v4 = *(_QWORD **)(a2 + 24);
  if ( *(_QWORD *)(a2 + 16) - (_QWORD)v4 <= 7u )
  {
    sub_16E7EE0(a2, "generic(", 8u);
  }
  else
  {
    *v4 = 0x28636972656E6567LL;
    *(_QWORD *)(a2 + 24) += 8LL;
  }
  sub_38CDBE0(*(_QWORD *)(a1 + 24), a2, a3, 0);
  result = *(_BYTE **)(a2 + 24);
  if ( *(_BYTE **)(a2 + 16) == result )
    return (_BYTE *)sub_16E7EE0(a2, ")", 1u);
  *result = 41;
  ++*(_QWORD *)(a2 + 24);
  return result;
}
