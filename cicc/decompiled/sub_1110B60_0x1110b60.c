// Function: sub_1110B60
// Address: 0x1110b60
//
__int64 __fastcall sub_1110B60(__int64 a1, __int64 a2)
{
  int v2; // eax
  unsigned int v3; // ebx

  v3 = *(_DWORD *)(a2 + 8);
  if ( v3 <= 0x40 )
  {
    LOBYTE(v2) = *(_QWORD *)a2 == 0;
  }
  else
  {
    v2 = sub_C444A0(a2);
    LOBYTE(v2) = v3 == v2;
  }
  return v2 ^ 1u;
}
