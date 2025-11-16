// Function: sub_8E3480
// Address: 0x8e3480
//
__int64 __fastcall sub_8E3480(__int64 a1, int a2, _DWORD *a3)
{
  __int64 result; // rax

  if ( *(char *)(a1 + 142) < 0 )
    result = sub_684B30(0x4EEu, a3);
  *(_DWORD *)(a1 + 136) = a2;
  *(_BYTE *)(a1 + 142) |= 0x80u;
  return result;
}
