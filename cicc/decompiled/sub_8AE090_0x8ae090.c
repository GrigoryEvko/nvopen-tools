// Function: sub_8AE090
// Address: 0x8ae090
//
_BOOL8 __fastcall sub_8AE090(__int64 a1, __int64 a2, _QWORD *a3)
{
  _BOOL8 result; // rax
  _QWORD *v5; // rbx

  if ( (unsigned __int8)(*(_BYTE *)(a1 + 140) - 9) > 2u )
    return 0;
  result = sub_89EDD0(a1, a2, a3);
  if ( !result )
  {
    if ( dword_4F077C4 == 2 )
    {
      if ( (unsigned int)sub_8D23B0(a1) )
        sub_8AE000(a1);
    }
    v5 = **(_QWORD ***)(a1 + 168);
    if ( v5 )
    {
      while ( !sub_89EDD0(v5[5], a2, a3) )
      {
        v5 = (_QWORD *)*v5;
        if ( !v5 )
          return 0;
      }
      return 1;
    }
    else
    {
      return 0;
    }
  }
  return result;
}
