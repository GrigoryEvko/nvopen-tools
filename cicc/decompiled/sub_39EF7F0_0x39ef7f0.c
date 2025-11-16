// Function: sub_39EF7F0
// Address: 0x39ef7f0
//
bool __fastcall sub_39EF7F0(__int64 a1)
{
  __int64 v1; // rax

  v1 = *(unsigned int *)(a1 + 120);
  if ( !(_DWORD)v1 )
    JUMPOUT(0x439316);
  return *(_DWORD *)(*(_QWORD *)(*(_QWORD *)(a1 + 112) + 32 * v1 - 32) + 36LL) != 0;
}
