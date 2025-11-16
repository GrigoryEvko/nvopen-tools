// Function: sub_D94040
// Address: 0xd94040
//
bool __fastcall sub_D94040(__int64 a1)
{
  unsigned int v1; // ebx

  v1 = *(_DWORD *)(a1 + 8);
  if ( v1 <= 0x40 )
    return *(_QWORD *)a1 == 1;
  else
    return v1 - 1 == (unsigned int)sub_C444A0(a1);
}
