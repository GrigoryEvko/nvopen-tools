// Function: sub_2D04F60
// Address: 0x2d04f60
//
__int64 __fastcall sub_2D04F60(int *a1, __int64 a2)
{
  unsigned int v2; // r8d

  v2 = 0;
  if ( *a1 >= 0 && a1[1] >= 0 && *(_QWORD *)a1 && a1[2] <= *(_DWORD *)(a2 + 8) && a1[3] <= (unsigned int)dword_50158A8 )
    LOBYTE(v2) = a1[4] <= (unsigned int)dword_50157C8;
  return v2;
}
