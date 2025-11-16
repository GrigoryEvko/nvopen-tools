// Function: sub_2EC8BE0
// Address: 0x2ec8be0
//
__int64 __fastcall sub_2EC8BE0(__int64 a1, _QWORD *a2, __int64 a3)
{
  _QWORD *v3; // r15
  _QWORD *v4; // rbx
  unsigned int v5; // r13d
  __int64 v6; // r12

  v3 = &a2[a3];
  if ( a2 == v3 )
  {
    return 0;
  }
  else
  {
    v4 = a2;
    v5 = 0;
    do
    {
      while ( 1 )
      {
        v6 = *v4;
        if ( *(_DWORD *)(a1 + 24) != 1 )
          break;
        if ( (*(_BYTE *)(v6 + 254) & 2) == 0 )
          sub_2F8F770(*v4);
        if ( v5 < *(_DWORD *)(v6 + 244) )
          v5 = *(_DWORD *)(v6 + 244);
        if ( v3 == ++v4 )
          return v5;
      }
      if ( (*(_BYTE *)(v6 + 254) & 1) == 0 )
        sub_2F8F5D0(*v4);
      if ( v5 < *(_DWORD *)(v6 + 240) )
        v5 = *(_DWORD *)(v6 + 240);
      ++v4;
    }
    while ( v3 != v4 );
  }
  return v5;
}
