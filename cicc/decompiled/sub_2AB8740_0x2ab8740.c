// Function: sub_2AB8740
// Address: 0x2ab8740
//
__int64 __fastcall sub_2AB8740(__int64 a1, __int64 a2)
{
  __int64 result; // rax

  result = *(_QWORD *)(a1 + 368);
  if ( !result )
    return sub_2AB8310(a1, a2);
  return result;
}
