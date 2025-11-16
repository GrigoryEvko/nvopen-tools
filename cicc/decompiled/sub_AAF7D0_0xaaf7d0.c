// Function: sub_AAF7D0
// Address: 0xaaf7d0
//
char __fastcall sub_AAF7D0(__int64 a1)
{
  unsigned int v1; // ebx
  char result; // al

  v1 = *(_DWORD *)(a1 + 8);
  if ( v1 <= 0x40 )
    return *(_QWORD *)a1 == *(_QWORD *)(a1 + 16) && *(_QWORD *)a1 == 0;
  result = sub_C43C50(a1, a1 + 16);
  if ( result )
    return v1 == (unsigned int)sub_C444A0(a1);
  return result;
}
