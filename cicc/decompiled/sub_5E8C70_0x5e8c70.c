// Function: sub_5E8C70
// Address: 0x5e8c70
//
__int64 __fastcall sub_5E8C70(__int64 a1, __int64 a2)
{
  __int64 v2; // rax
  __int64 v3; // r8
  __int64 v5; // rdx
  __int64 v6; // rcx
  __int64 v7; // rbx

  v2 = *(_QWORD *)(a1 + 152);
  LODWORD(v3) = 0;
  if ( !v2 )
    return (unsigned int)v3;
  if ( (*(_BYTE *)(v2 + 29) & 0x20) != 0 )
    return (unsigned int)v3;
  v3 = 1;
  if ( *(_QWORD *)(v2 + 112) )
    return (unsigned int)v3;
  v5 = *(_QWORD *)(v2 + 144);
  v6 = dword_4D04824;
  if ( !dword_4D04824 )
  {
    while ( v5 )
    {
      if ( *(char *)(v5 + 192) >= 0 )
      {
        LODWORD(v3) = 1;
        return (unsigned int)v3;
      }
      v5 = *(_QWORD *)(v5 + 112);
    }
    goto LABEL_7;
  }
  if ( v5 )
    return (unsigned int)v3;
LABEL_7:
  v7 = *(_QWORD *)(v2 + 104);
  if ( v7 )
  {
    while ( (unsigned __int8)(*(_BYTE *)(v7 + 140) - 9) > 2u
         || !(unsigned int)sub_5E8C70(*(_QWORD *)(v7 + 168), a2, v5, v6, v3) )
    {
      v7 = *(_QWORD *)(v7 + 112);
      if ( !v7 )
        return 0;
    }
    return 1;
  }
  else
  {
    return 0;
  }
}
