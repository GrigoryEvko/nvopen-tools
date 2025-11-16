// Function: sub_C26F20
// Address: 0xc26f20
//
__int64 __fastcall sub_C26F20(_QWORD *a1)
{
  __int64 v1; // rax
  __int64 result; // rax

  v1 = a1[9];
  a1[26] = *(_QWORD *)(v1 + 8);
  a1[27] = *(_QWORD *)(v1 + 16);
  result = sub_C22170(a1);
  if ( !(_DWORD)result )
  {
    LODWORD(result) = sub_C26EA0(a1);
    if ( (_DWORD)result )
    {
      return (unsigned int)result;
    }
    else
    {
      sub_C1AFD0();
      return 0;
    }
  }
  return result;
}
