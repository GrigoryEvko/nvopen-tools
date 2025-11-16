// Function: sub_9867B0
// Address: 0x9867b0
//
bool __fastcall sub_9867B0(__int64 a1)
{
  unsigned int v1; // ebx

  v1 = *(_DWORD *)(a1 + 8);
  if ( v1 <= 0x40 )
    return *(_QWORD *)a1 == 0;
  else
    return (unsigned int)sub_C444A0(a1) == v1;
}
