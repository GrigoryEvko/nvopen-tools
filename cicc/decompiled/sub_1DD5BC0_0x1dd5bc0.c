// Function: sub_1DD5BC0
// Address: 0x1dd5bc0
//
__int64 __fastcall sub_1DD5BC0(__int64 a1, __int64 a2)
{
  __int64 result; // rax

  result = sub_1E15F70(a2);
  if ( result )
    result = sub_1E15BD0(a2, *(_QWORD *)(result + 40));
  *(_QWORD *)(a2 + 24) = 0;
  return result;
}
