// Function: sub_15520E0
// Address: 0x15520e0
//
_BYTE *__fastcall sub_15520E0(__int64 *a1, __int64 *a2, char a3)
{
  __int64 v3; // r13
  __int64 v6; // rdi
  _BYTE *v7; // rax

  if ( !a2 )
    return (_BYTE *)sub_1263B40(*a1, "<null operand!>");
  v3 = (__int64)(a1 + 5);
  if ( a3 )
  {
    sub_154DAA0((__int64)(a1 + 5), *a2, *a1);
    v6 = *a1;
    v7 = *(_BYTE **)(*a1 + 24);
    if ( (unsigned __int64)v7 >= *(_QWORD *)(*a1 + 16) )
    {
      sub_16E7DE0(v6, 32);
    }
    else
    {
      *(_QWORD *)(v6 + 24) = v7 + 1;
      *v7 = 32;
    }
  }
  return sub_1550E20(*a1, (__int64)a2, v3, a1[4], a1[1]);
}
