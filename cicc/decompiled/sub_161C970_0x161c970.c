// Function: sub_161C970
// Address: 0x161c970
//
__int64 __fastcall sub_161C970(_QWORD *a1, _QWORD *a2)
{
  unsigned int v2; // edx
  unsigned int v3; // ecx
  __int64 result; // rax

  v2 = *(_DWORD *)a1;
  v3 = *(_DWORD *)a2;
  result = 0xFFFFFFFFLL;
  if ( *(_DWORD *)a2 <= *(_DWORD *)a1 && (v3 != v2 || a1[1] >= a2[1]) )
  {
    result = 1;
    if ( v3 >= v2 )
      return a2[1] < a1[1];
  }
  return result;
}
