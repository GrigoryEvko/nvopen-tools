// Function: sub_14237C0
// Address: 0x14237c0
//
_BYTE *__fastcall sub_14237C0(__int64 a1, __int64 a2, __int64 a3)
{
  _BYTE *result; // rax
  _WORD *v5; // rdx
  __int64 v6; // r13

  result = (_BYTE *)sub_14228C0(*(_QWORD *)(a1 + 8), a2);
  if ( result )
  {
    v5 = *(_WORD **)(a3 + 24);
    v6 = (__int64)result;
    if ( *(_QWORD *)(a3 + 16) - (_QWORD)v5 <= 1u )
    {
      a3 = sub_16E7EE0(a3, "; ", 2);
    }
    else
    {
      *v5 = 8251;
      *(_QWORD *)(a3 + 24) += 2LL;
    }
    sub_14236E0(v6, a3);
    result = *(_BYTE **)(a3 + 24);
    if ( *(_BYTE **)(a3 + 16) == result )
    {
      return (_BYTE *)sub_16E7EE0(a3, "\n", 1);
    }
    else
    {
      *result = 10;
      ++*(_QWORD *)(a3 + 24);
    }
  }
  return result;
}
