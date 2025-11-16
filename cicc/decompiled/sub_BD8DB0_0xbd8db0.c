// Function: sub_BD8DB0
// Address: 0xbd8db0
//
__int64 __fastcall sub_BD8DB0(_QWORD *a1, _QWORD *a2)
{
  __int64 result; // rax

  result = *a2 < *a1;
  if ( *a2 > *a1 )
    return 0xFFFFFFFFLL;
  return result;
}
