// Function: sub_7E1F90
// Address: 0x7e1f90
//
__int64 __fastcall sub_7E1F90(__int64 a1)
{
  __int64 i; // r12
  __int64 result; // rax
  __int64 v3; // rax

  for ( i = a1; *(_BYTE *)(i + 140) == 12; i = *(_QWORD *)(i + 160) )
    ;
  result = 1;
  if ( qword_4F18A00 != i )
  {
    result = sub_8D3D10(i);
    if ( (_DWORD)result )
    {
      v3 = sub_8D4870(i);
      return (unsigned int)sub_8D2310(v3) != 0;
    }
  }
  return result;
}
