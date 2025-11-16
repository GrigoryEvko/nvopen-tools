// Function: sub_158A0B0
// Address: 0x158a0b0
//
char __fastcall sub_158A0B0(__int64 a1)
{
  unsigned int v1; // ebx
  char result; // al

  v1 = *(_DWORD *)(a1 + 8);
  if ( v1 <= 0x40 )
    return *(_QWORD *)a1 == *(_QWORD *)(a1 + 16) && 0xFFFFFFFFFFFFFFFFLL >> (64 - (unsigned __int8)v1) == *(_QWORD *)a1;
  result = sub_16A5220(a1, a1 + 16);
  if ( result )
    return v1 == (unsigned int)sub_16A58F0(a1);
  return result;
}
