// Function: sub_C8B3D0
// Address: 0xc8b3d0
//
__int64 sub_C8B3D0()
{
  __int64 result; // rax

  result = sub_22077B0(32);
  if ( result )
  {
    *(_QWORD *)(result + 8) = 0;
    *(_QWORD *)result = result + 16;
    *(_BYTE *)(result + 16) = 0;
  }
  return result;
}
