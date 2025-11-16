// Function: sub_14ACA50
// Address: 0x14aca50
//
__int64 __fastcall sub_14ACA50(__int64 a1, unsigned int a2)
{
  unsigned int v2; // r12d
  __int64 v4; // rax
  __int64 v5; // rbx
  __int64 v6; // rdi
  unsigned int v7; // ebx

  if ( (*(_DWORD *)(a1 + 20) & 0xFFFFFFF) == 3 )
  {
    v4 = sub_16348C0(a1);
    if ( *(_BYTE *)(v4 + 8) == 14 )
    {
      v2 = sub_1642F90(*(_QWORD *)(v4 + 24), a2);
      if ( (_BYTE)v2 )
      {
        v5 = (*(_BYTE *)(a1 + 23) & 0x40) != 0 ? *(_QWORD *)(a1 - 8) : a1 - 24LL * (*(_DWORD *)(a1 + 20) & 0xFFFFFFF);
        v6 = *(_QWORD *)(v5 + 24);
        if ( *(_BYTE *)(v6 + 16) == 13 )
        {
          v7 = *(_DWORD *)(v6 + 32);
          if ( v7 <= 0x40 )
          {
            if ( !*(_QWORD *)(v6 + 24) )
              return v2;
          }
          else if ( v7 == (unsigned int)sub_16A57B0(v6 + 24) )
          {
            return v2;
          }
        }
      }
    }
  }
  return 0;
}
