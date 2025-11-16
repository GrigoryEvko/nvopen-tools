// Function: sub_16E3D10
// Address: 0x16e3d10
//
__int64 __fastcall sub_16E3D10(__int64 a1, __int64 a2, __int64 a3, __int64 a4, int a5, int a6)
{
  __int64 result; // rax

  result = *(unsigned int *)(a1 + 40);
  if ( (unsigned int)result >= *(_DWORD *)(a1 + 44) )
  {
    sub_16CD150(a1 + 32, (const void *)(a1 + 48), 0, 4, a5, a6);
    result = *(unsigned int *)(a1 + 40);
  }
  *(_DWORD *)(*(_QWORD *)(a1 + 32) + 4 * result) = 2;
  *(_BYTE *)(a1 + 95) = 1;
  ++*(_DWORD *)(a1 + 40);
  return result;
}
