// Function: sub_F8E3D0
// Address: 0xf8e3d0
//
__int64 __fastcall sub_F8E3D0(_QWORD *a1, _QWORD *a2)
{
  __int64 result; // rax

  result = *a2 < *a1;
  if ( *a2 > *a1 )
    return 0xFFFFFFFFLL;
  return result;
}
