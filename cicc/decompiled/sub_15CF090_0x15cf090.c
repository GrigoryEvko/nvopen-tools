// Function: sub_15CF090
// Address: 0x15cf090
//
_QWORD *__fastcall sub_15CF090(_QWORD *a1, __int64 a2, __int64 *a3)
{
  _QWORD *v3; // rax
  _QWORD *v4; // r9
  _QWORD *v5; // r8
  _QWORD *i; // rax

  v3 = sub_15CBCA0(a1, a2, a3);
  v5 = v3;
  if ( (_QWORD *)a2 != v3 )
  {
    for ( i = v3 + 1; (_QWORD *)a2 != i; ++i )
    {
      if ( *i != *v4 )
        *v5++ = *i;
    }
  }
  return v5;
}
