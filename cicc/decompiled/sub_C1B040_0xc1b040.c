// Function: sub_C1B040
// Address: 0xc1b040
//
__int64 __fastcall sub_C1B040(__int64 a1)
{
  unsigned __int8 v1; // al
  int v2; // ebx
  unsigned __int8 **v3; // rdi

  v1 = *(_BYTE *)(a1 - 16);
  v2 = *(_DWORD *)(a1 + 4);
  if ( (v1 & 2) != 0 )
    v3 = *(unsigned __int8 ***)(a1 - 32);
  else
    v3 = (unsigned __int8 **)(a1 - 16 - 8LL * ((v1 >> 2) & 0xF));
  return (unsigned __int16)(v2 - *((_WORD *)sub_AF34D0(*v3) + 8));
}
