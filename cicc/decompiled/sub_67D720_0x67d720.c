// Function: sub_67D720
// Address: 0x67d720
//
__int64 __fastcall sub_67D720(_QWORD *a1, int a2)
{
  __int64 result; // rax
  __int64 v3; // rdx
  __int64 v4; // rdx

  result = sub_67B9F0();
  *(_DWORD *)result = 1;
  *(_DWORD *)(result + 176) = a2;
  v3 = a1[12];
  *(_QWORD *)(result + 16) = a1;
  *(_QWORD *)(result + 96) = v3;
  if ( !a1[3] )
    a1[3] = result;
  v4 = a1[4];
  if ( v4 )
    *(_QWORD *)(v4 + 8) = result;
  a1[4] = result;
  return result;
}
