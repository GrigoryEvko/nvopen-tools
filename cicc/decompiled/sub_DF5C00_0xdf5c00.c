// Function: sub_DF5C00
// Address: 0xdf5c00
//
__int64 __fastcall sub_DF5C00(__int64 a1, __int64 a2, _DWORD *a3)
{
  *a3 = 0;
  return ((*(_DWORD *)(a2 + 4) & 0x7FFFFFFu) >> 1) - 1;
}
