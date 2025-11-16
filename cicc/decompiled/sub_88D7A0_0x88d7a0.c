// Function: sub_88D7A0
// Address: 0x88d7a0
//
__int64 __fastcall sub_88D7A0(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  unsigned __int8 v6; // al
  __int64 result; // rax
  __int64 *v8; // rax
  __int64 v9; // rdx
  __int64 v10; // rax
  __int64 v11; // rdi

  v6 = *(_BYTE *)(a1 + 8);
  if ( v6 == 2 )
  {
    v8 = *(__int64 **)(a1 + 32);
    if ( !v8 )
      return 1;
    v9 = *v8;
    v10 = *(_QWORD *)(*v8 + 88);
    if ( (*(_BYTE *)(v10 + 160) & 2) != 0 || (*(_BYTE *)(v10 + 266) & 1) != 0 )
    {
      return 1;
    }
    else
    {
      result = 0;
      if ( (*(_BYTE *)(v9 + 81) & 0x10) != 0 )
        return sub_8DC060(*(_QWORD *)(v9 + 64));
    }
  }
  else
  {
    if ( v6 > 2u )
    {
      if ( v6 != 3 )
        sub_721090();
      return 0;
    }
    if ( v6 )
    {
      if ( *(_QWORD *)(a1 + 48) )
        return sub_697860(*(_QWORD *)(a1 + 48));
      if ( (*(_BYTE *)(a1 + 24) & 1) == 0 )
      {
        v11 = *(_QWORD *)(a1 + 32);
        result = 1;
        if ( v11 )
        {
          if ( *(_BYTE *)(v11 + 173) != 12 )
            return sub_7322D0(v11, a2, a3, a4, 0, a6);
        }
        return result;
      }
      return 0;
    }
    result = 1;
    if ( *(_QWORD *)(a1 + 32) )
      return (unsigned int)sub_8DC100() != 0;
  }
  return result;
}
