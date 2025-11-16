// Function: sub_2BDB750
// Address: 0x2bdb750
//
bool __fastcall sub_2BDB750(_QWORD *a1, _BYTE *a2)
{
  return (*(_QWORD *)(*a1 + 8 * ((unsigned __int64)(unsigned __int8)*a2 >> 6) + 128) & (1LL << *a2)) != 0;
}
