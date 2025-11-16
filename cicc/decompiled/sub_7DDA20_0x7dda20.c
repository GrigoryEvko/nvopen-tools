// Function: sub_7DDA20
// Address: 0x7dda20
//
__int64 __fastcall sub_7DDA20(__int64 a1)
{
  __int64 result; // rax

  result = *(_QWORD *)(a1 + 152);
  if ( !result )
    return sub_7DC650(a1);
  return result;
}
