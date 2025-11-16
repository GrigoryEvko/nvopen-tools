// Function: sub_324A2D0
// Address: 0x324a2d0
//
__int64 __fastcall sub_324A2D0(__int64 *a1, __int64 a2, __int64 a3, char a4)
{
  unsigned int v4; // eax
  __int64 v5; // r8
  __int64 v7; // rcx

  v4 = *(_DWORD *)(a3 + 8);
  if ( v4 <= 0x40 )
  {
    v5 = *(_QWORD *)a3;
    if ( !a4 )
    {
      v7 = 0;
      if ( !v4 )
        return sub_3249E90(a1, a2, a4, v7);
      v5 = v5 << (64 - (unsigned __int8)v4) >> (64 - (unsigned __int8)v4);
    }
    v7 = v5;
    return sub_3249E90(a1, a2, a4, v7);
  }
  return sub_324A160(a1, a2, 28, (__int64 *)a3);
}
