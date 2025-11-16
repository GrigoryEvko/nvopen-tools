// Function: sub_AA50C0
// Address: 0xaa50c0
//
__int64 __fastcall sub_AA50C0(__int64 a1, __int64 a2)
{
  __int64 v2; // r12
  __int64 v3; // rbx
  char v4; // al
  __int64 v5; // rdi
  __int64 v7; // r14

  v2 = a1 + 48;
  v3 = *(_QWORD *)(a1 + 56);
  if ( v3 != a1 + 48 )
  {
    do
    {
      if ( !v3 )
        BUG();
      v4 = *(_BYTE *)(v3 - 24);
      v5 = v3 - 24;
      if ( v4 != 84 )
      {
        if ( v4 == 85 )
        {
          v7 = *(_QWORD *)(v3 - 56);
          if ( (!v7
             || *(_BYTE *)v7
             || *(_QWORD *)(v7 + 24) != *(_QWORD *)(v3 + 56)
             || (*(_BYTE *)(v7 + 33) & 0x20) == 0
             || (unsigned int)(*(_DWORD *)(v7 + 36) - 68) > 3)
            && !(unsigned __int8)sub_B46A10(v5, a2)
            && (!(_BYTE)a2
             || !v7
             || *(_BYTE *)v7
             || *(_QWORD *)(v7 + 24) != *(_QWORD *)(v3 + 56)
             || (*(_BYTE *)(v7 + 33) & 0x20) == 0
             || *(_DWORD *)(v7 + 36) != 291) )
          {
            return v3;
          }
        }
        else if ( !(unsigned __int8)sub_B46A10(v5, a2) )
        {
          return v3;
        }
      }
      v3 = *(_QWORD *)(v3 + 8);
    }
    while ( v2 != v3 );
  }
  return v2;
}
