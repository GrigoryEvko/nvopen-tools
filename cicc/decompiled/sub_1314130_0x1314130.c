// Function: sub_1314130
// Address: 0x1314130
//
__int64 __fastcall sub_1314130(_DWORD *a1, unsigned __int64 *a2)
{
  unsigned __int64 v2; // rax
  unsigned __int64 v3; // rcx

  v2 = 0x5851F42D4C957F2DLL * *a2 + 0x14057B7EF767814FLL;
  *a2 = v2;
  v3 = (int)a1[1] * (unsigned __int64)byte_4287520[v2 >> 58];
  *a1 = v3 / 0x3D;
  return 1;
}
