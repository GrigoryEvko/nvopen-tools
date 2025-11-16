// Function: sub_2AAE030
// Address: 0x2aae030
//
__int64 __fastcall sub_2AAE030(__int64 *a1, __int64 a2)
{
  __int64 v3; // [rsp+0h] [rbp-8h]

  BYTE4(v3) = *(_BYTE *)(a2 + 8) == 18;
  LODWORD(v3) = *(_DWORD *)(a2 + 32);
  return sub_BCE1B0(a1, v3);
}
