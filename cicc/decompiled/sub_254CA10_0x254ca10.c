// Function: sub_254CA10
// Address: 0x254ca10
//
__int64 __fastcall sub_254CA10(__int64 a1, signed int a2)
{
  int v2; // eax
  unsigned int v3; // eax

  v2 = *(_DWORD *)(a1 + 16);
  if ( !v2 )
  {
    if ( !sub_B491E0(*(_QWORD *)a1) )
    {
      v3 = sub_A17190(*(unsigned __int8 **)a1);
      goto LABEL_3;
    }
    v2 = *(_DWORD *)(a1 + 16);
  }
  v3 = v2 - 1;
LABEL_3:
  if ( a2 < v3 )
  {
    if ( *(_DWORD *)(a1 + 16) || sub_B491E0(*(_QWORD *)a1) )
      a2 = *(_DWORD *)(*(_QWORD *)(a1 + 8) + 4LL * (unsigned int)(a2 + 1));
    if ( a2 >= 0 )
      return sub_254C9B0(*(_QWORD *)a1, a2);
  }
  nullsub_1518();
  return 0;
}
