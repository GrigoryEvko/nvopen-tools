// Function: sub_72F070
// Address: 0x72f070
//
__int64 __fastcall sub_72F070(__int64 a1)
{
  __int64 result; // rax
  __int64 v3; // rdi
  __int64 v4; // r8
  __int64 v5; // rax
  _QWORD *v6; // rax
  __int64 v7; // rax

  result = *(_QWORD *)(a1 + 40);
  if ( (*(_BYTE *)(a1 + 89) & 2) != 0 )
  {
    v3 = *(_QWORD *)(a1 + 48);
    if ( (*(_BYTE *)(v3 + 197) & 0x60) != 0 && (v4 = *(_QWORD *)(v3 + 128)) != 0 )
    {
      if ( *(_DWORD *)(v4 + 160) )
      {
        v5 = sub_72B840(*(_QWORD *)(v3 + 128));
        if ( !v5 )
          return 0;
        v6 = *(_QWORD **)(v5 + 224);
        if ( !v6 )
          return 0;
        while ( a1 != v6[3] )
        {
          v6 = (_QWORD *)*v6;
          if ( !v6 )
            return 0;
        }
        return v6[1];
      }
    }
    else if ( *(_DWORD *)(v3 + 160) )
    {
      v7 = sub_72B840(v3);
      if ( !v7 )
        return 0;
      v6 = *(_QWORD **)(v7 + 224);
      if ( !v6 )
        return 0;
      while ( a1 != v6[3] )
      {
        v6 = (_QWORD *)*v6;
        if ( !v6 )
          return 0;
      }
      return v6[1];
    }
  }
  return result;
}
