// Function: sub_B4CE70
// Address: 0xb4ce70
//
__int64 __fastcall sub_B4CE70(__int64 a1)
{
  int v1; // eax
  __int64 v2; // rdi
  unsigned int v3; // ebx

  v2 = *(_QWORD *)(a1 - 32);
  if ( *(_BYTE *)v2 != 17 )
    return 1;
  v3 = *(_DWORD *)(v2 + 32);
  if ( v3 <= 0x40 )
  {
    LOBYTE(v1) = *(_QWORD *)(v2 + 24) == 1;
  }
  else
  {
    v1 = sub_C444A0(v2 + 24);
    LOBYTE(v1) = v3 - 1 == v1;
  }
  return v1 ^ 1u;
}
