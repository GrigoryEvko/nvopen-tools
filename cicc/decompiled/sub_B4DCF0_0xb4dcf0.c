// Function: sub_B4DCF0
// Address: 0xb4dcf0
//
__int64 __fastcall sub_B4DCF0(__int64 a1)
{
  unsigned int v1; // r12d
  unsigned int v3; // ebx
  __int64 v4; // rdi
  unsigned int v5; // r14d

  v1 = *(_DWORD *)(a1 + 4) & 0x7FFFFFF;
  if ( v1 == 1 )
    return 1;
  v3 = 1;
  while ( 1 )
  {
    v4 = *(_QWORD *)(a1 + 32 * (v3 - (unsigned __int64)v1));
    if ( *(_BYTE *)v4 != 17 )
      break;
    v5 = *(_DWORD *)(v4 + 32);
    if ( v5 <= 0x40 )
    {
      if ( *(_QWORD *)(v4 + 24) )
        return 0;
    }
    else if ( v5 != (unsigned int)sub_C444A0(v4 + 24) )
    {
      return 0;
    }
    if ( v1 == ++v3 )
      return 1;
  }
  return 0;
}
