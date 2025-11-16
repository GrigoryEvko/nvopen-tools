// Function: sub_161E7C0
// Address: 0x161e7c0
//
unsigned __int64 __fastcall sub_161E7C0(__int64 a1, __int64 a2)
{
  unsigned __int64 result; // rax

  result = sub_161E760((unsigned __int8 *)a2);
  if ( result )
    return sub_161E590(result, a1);
  if ( *(_BYTE *)a2 == 3 )
    *(_QWORD *)(a2 + 8) = 0;
  return result;
}
