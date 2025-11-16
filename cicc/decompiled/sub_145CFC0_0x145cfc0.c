// Function: sub_145CFC0
// Address: 0x145cfc0
//
__int64 __fastcall sub_145CFC0(__int64 a1, __int64 a2, __int64 a3)
{
  __int64 v3; // rbx
  unsigned int v4; // r14d
  __int64 v5; // rax
  __int64 v7; // rax

  if ( !*(_WORD *)(a3 + 24) )
  {
    v3 = *(_QWORD *)(a3 + 32);
    v4 = *(_DWORD *)(v3 + 32);
    if ( v4 <= 0x40 )
    {
      if ( *(_QWORD *)(v3 + 24) )
        goto LABEL_4;
    }
    else if ( v4 != (unsigned int)sub_16A57B0(v3 + 24) )
    {
LABEL_4:
      v5 = sub_145CF80(a2, *(_QWORD *)v3, 0, 0);
      sub_14573F0(a1, v5);
      return a1;
    }
  }
  v7 = sub_1456E90(a2);
  sub_14573F0(a1, v7);
  return a1;
}
