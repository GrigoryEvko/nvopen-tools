// Function: sub_2BDB7B0
// Address: 0x2bdb7b0
//
bool __fastcall sub_2BDB7B0(_QWORD *a1, _BYTE *a2)
{
  return (*(_QWORD *)(*a1 + 8 * ((unsigned __int64)(unsigned __int8)*a2 >> 6) + 128) & (1LL << *a2)) != 0;
}
