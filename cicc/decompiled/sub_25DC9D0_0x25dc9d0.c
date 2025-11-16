// Function: sub_25DC9D0
// Address: 0x25dc9d0
//
bool __fastcall sub_25DC9D0(__int64 a1)
{
  __int64 v1; // rax
  __int64 v2; // rdx
  __int64 v3; // r12
  unsigned int v5; // ebx

  v1 = *(_DWORD *)(a1 + 4) & 0x7FFFFFF;
  v2 = *(_QWORD *)(a1 + 32 * (3 - v1));
  if ( *(_BYTE *)v2 != 17 )
    return 0;
  v3 = *(_QWORD *)(a1 - 32 * v1);
  if ( *(_BYTE *)v3 != 60 )
    return 0;
  v5 = *(_DWORD *)(v2 + 32);
  if ( v5 <= 0x40 )
  {
    if ( *(_QWORD *)(v2 + 24) == 1 )
      return 0;
  }
  else if ( (unsigned int)sub_C444A0(v2 + 24) == v5 - 1 )
  {
    return 0;
  }
  if ( !sub_B4D040(v3) )
    return 0;
  return *(_BYTE *)(*(_QWORD *)(v3 + 72) + 8LL) == 16;
}
