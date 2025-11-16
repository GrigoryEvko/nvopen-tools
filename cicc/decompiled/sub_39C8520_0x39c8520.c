// Function: sub_39C8520
// Address: 0x39c8520
//
__int64 __fastcall sub_39C8520(_QWORD *a1)
{
  unsigned int v1; // r12d
  __int64 v2; // rax

  v1 = *(unsigned __int8 *)(a1[10] + 50LL);
  if ( !(_BYTE)v1 )
  {
    v2 = a1[25];
    if ( *(_DWORD *)(v2 + 6584) == 1 && *(_BYTE *)(v2 + 4500) && !sub_39C84F0(a1) )
      LOBYTE(v1) = *(_DWORD *)(a1[10] + 36LL) != 3;
  }
  return v1;
}
