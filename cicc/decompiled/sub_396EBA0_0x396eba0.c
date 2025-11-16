// Function: sub_396EBA0
// Address: 0x396eba0
//
__int64 __fastcall sub_396EBA0(__int64 a1)
{
  unsigned int v1; // r12d
  __int64 v2; // rax
  __int64 v4; // rbx

  v1 = 0;
  v2 = *(_QWORD *)(a1 + 240);
  if ( *(_DWORD *)(v2 + 348) == 4 )
  {
    LOBYTE(v1) = *(_DWORD *)(v2 + 352) != 6 && *(_DWORD *)(v2 + 352) != 0;
    if ( (_BYTE)v1 )
    {
      v4 = **(_QWORD **)(a1 + 264);
      if ( !(unsigned __int8)sub_1560180(v4 + 112, 56) )
      {
        if ( (unsigned __int8)sub_1560180(v4 + 112, 30) )
          return (*(_WORD *)(v4 + 18) >> 3) & 1;
      }
    }
  }
  return v1;
}
