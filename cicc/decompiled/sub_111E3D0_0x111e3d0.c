// Function: sub_111E3D0
// Address: 0x111e3d0
//
__int64 __fastcall sub_111E3D0(_QWORD **a1, _BYTE *a2)
{
  __int64 result; // rax
  __int64 v3; // rdx

  result = 0;
  if ( *a2 == 75 )
  {
    v3 = *((_QWORD *)a2 - 4);
    if ( v3 )
    {
      **a1 = v3;
      return 1;
    }
  }
  return result;
}
