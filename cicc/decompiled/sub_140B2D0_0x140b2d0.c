// Function: sub_140B2D0
// Address: 0x140b2d0
//
__int64 __fastcall sub_140B2D0(_QWORD *a1)
{
  __int64 result; // rax

  result = sub_140B250(a1);
  if ( result )
    return *(_QWORD *)(result + 24);
  return result;
}
