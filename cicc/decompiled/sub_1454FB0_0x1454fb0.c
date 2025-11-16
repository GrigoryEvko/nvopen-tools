// Function: sub_1454FB0
// Address: 0x1454fb0
//
bool __fastcall sub_1454FB0(__int64 a1)
{
  unsigned int v1; // ebx

  v1 = *(_DWORD *)(a1 + 8);
  if ( v1 <= 0x40 )
    return 0xFFFFFFFFFFFFFFFFLL >> (64 - (unsigned __int8)v1) == *(_QWORD *)a1;
  else
    return (unsigned int)sub_16A58F0(a1) == v1;
}
