// Function: sub_98AB00
// Address: 0x98ab00
//
__int64 __fastcall sub_98AB00(__int64 a1, __int64 a2, __int64 a3)
{
  unsigned int v3; // r12d
  __int64 v5; // rax
  __int64 v6; // rdi
  unsigned int v7; // ebx

  if ( (*(_DWORD *)(a1 + 4) & 0x7FFFFFF) == 3 )
  {
    v5 = sub_BB5290(a1, a2, a3);
    if ( *(_BYTE *)(v5 + 8) == 16 )
    {
      v3 = sub_BCAC40(*(_QWORD *)(v5 + 24), (unsigned int)a2);
      if ( (_BYTE)v3 )
      {
        v6 = *(_QWORD *)(a1 + 32 * (1LL - (*(_DWORD *)(a1 + 4) & 0x7FFFFFF)));
        if ( *(_BYTE *)v6 == 17 )
        {
          v7 = *(_DWORD *)(v6 + 32);
          if ( v7 <= 0x40 )
          {
            if ( !*(_QWORD *)(v6 + 24) )
              return v3;
          }
          else if ( v7 == (unsigned int)sub_C444A0(v6 + 24) )
          {
            return v3;
          }
        }
      }
    }
  }
  return 0;
}
