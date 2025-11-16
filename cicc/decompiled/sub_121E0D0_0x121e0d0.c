// Function: sub_121E0D0
// Address: 0x121e0d0
//
_BYTE *__fastcall sub_121E0D0(__int64 a1, unsigned int a2, unsigned __int64 a3)
{
  _QWORD *v4; // rax
  __int64 v5; // rax
  _BYTE *result; // rax

  v4 = (_QWORD *)sub_B2BE50(*(_QWORD *)(a1 + 8));
  v5 = sub_BCB130(v4);
  result = (_BYTE *)sub_121DDC0(a1, a2, v5, a3);
  if ( result )
  {
    if ( *result != 23 )
      return 0;
  }
  return result;
}
