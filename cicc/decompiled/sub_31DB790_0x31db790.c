// Function: sub_31DB790
// Address: 0x31db790
//
__int64 __fastcall sub_31DB790(__int64 a1)
{
  unsigned int v1; // r12d
  __int64 v2; // rax
  __int64 v4; // rbx

  v1 = 0;
  v2 = *(_QWORD *)(a1 + 208);
  if ( *(_DWORD *)(v2 + 336) == 4 )
  {
    LOBYTE(v1) = *(_DWORD *)(v2 + 344) != 6 && *(_DWORD *)(v2 + 344) != 0;
    if ( (_BYTE)v1 )
    {
      v4 = **(_QWORD **)(a1 + 232);
      if ( !(unsigned int)sub_A746B0((_QWORD *)(v4 + 120)) )
      {
        if ( (unsigned __int8)sub_B2D610(v4, 41) )
          return (*(_WORD *)(v4 + 2) >> 3) & 1;
      }
    }
  }
  return v1;
}
