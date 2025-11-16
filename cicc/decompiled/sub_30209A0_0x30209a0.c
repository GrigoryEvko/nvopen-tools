// Function: sub_30209A0
// Address: 0x30209a0
//
__int64 __fastcall sub_30209A0(__int64 a1, __int64 *a2)
{
  unsigned __int8 v2; // al
  const char *v3; // rax
  __int64 v4; // rdx
  __int64 v5; // rax
  __int64 v7; // r12

  v2 = *(_BYTE *)a1;
  if ( *(_BYTE *)a1 == 3 )
  {
    v3 = sub_BD5D20(a1);
    if ( v4 == 9 && *(_QWORD *)v3 == 0x6573752E6D766C6CLL && v3[8] == 100 )
      return 1;
    v2 = *(_BYTE *)a1;
  }
  if ( v2 <= 0x1Cu )
  {
    v7 = *(_QWORD *)(a1 + 16);
    if ( v7 )
    {
      while ( (unsigned __int8)sub_30209A0(*(_QWORD *)(v7 + 24), a2) )
      {
        v7 = *(_QWORD *)(v7 + 8);
        if ( !v7 )
          return 1;
      }
      return 0;
    }
  }
  else
  {
    v5 = sub_B43CB0(a1);
    if ( !v5 || *a2 != v5 && *a2 )
      return 0;
    *a2 = v5;
  }
  return 1;
}
