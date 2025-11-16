// Function: sub_AD6CA0
// Address: 0xad6ca0
//
__int64 __fastcall sub_AD6CA0(__int64 a1)
{
  __int64 v1; // rax
  int v2; // r13d
  unsigned int v3; // ebx

  if ( (unsigned __int8)(*(_BYTE *)a1 - 17) <= 1u )
    return 0;
  v1 = *(_QWORD *)(a1 + 8);
  if ( *(_BYTE *)(v1 + 8) != 17 )
    return 0;
  v2 = *(_DWORD *)(v1 + 32);
  if ( !v2 )
    return 0;
  v3 = 0;
  while ( *(_BYTE *)sub_AD69F0((unsigned __int8 *)a1, v3) != 5 )
  {
    if ( ++v3 == v2 )
      return 0;
  }
  return 1;
}
