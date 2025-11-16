// Function: sub_C33CE0
// Address: 0xc33ce0
//
bool __fastcall sub_C33CE0(__int64 a1)
{
  return (*(_BYTE *)(a1 + 20) & 7) == 2 && *(_DWORD *)(a1 + 16) == *(_DWORD *)(*(_QWORD *)a1 + 4LL) && sub_C33C40(a1);
}
