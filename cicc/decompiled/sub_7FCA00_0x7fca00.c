// Function: sub_7FCA00
// Address: 0x7fca00
//
__int64 __fastcall sub_7FCA00(_QWORD *a1)
{
  __int64 result; // rax
  __int64 v2; // rdx

  if ( a1 )
  {
    *a1 = *(_QWORD *)dword_4D03F38;
    a1[1] = *(_QWORD *)dword_4D03F38;
  }
  result = unk_4D03EB0;
  if ( unk_4D03EB0 )
  {
    do
    {
      v2 = result;
      result = *(_QWORD *)(result + 16);
    }
    while ( result );
    *(_QWORD *)(v2 + 16) = a1;
    a1[2] = 0;
  }
  else
  {
    unk_4D03EB0 = a1;
    a1[2] = 0;
  }
  return result;
}
