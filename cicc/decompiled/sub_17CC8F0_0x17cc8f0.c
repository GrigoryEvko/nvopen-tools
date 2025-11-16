// Function: sub_17CC8F0
// Address: 0x17cc8f0
//
__int64 __fastcall sub_17CC8F0(__int64 a1, __int64 a2)
{
  __int64 v2; // r13
  __int64 v3; // r12
  __int64 v4; // rbx
  __int64 v5; // rsi
  char v6; // al
  __int64 v7; // rax

  v2 = a1 + 40;
  v3 = a2;
  if ( a1 + 40 != a2 )
  {
    v4 = a2;
    do
    {
      while ( 1 )
      {
        if ( !v4 )
          BUG();
        v6 = *(_BYTE *)(v4 - 8);
        if ( v6 != 53 )
          break;
        if ( (unsigned __int8)sub_15F8F00(v4 - 24) )
          goto LABEL_4;
LABEL_8:
        v4 = *(_QWORD *)(v4 + 8);
        if ( v2 == v4 )
          return v3;
      }
      if ( v6 != 78 )
        goto LABEL_8;
      v7 = *(_QWORD *)(v4 - 48);
      if ( *(_BYTE *)(v7 + 16) || (*(_BYTE *)(v7 + 33) & 0x20) == 0 )
        goto LABEL_8;
      if ( *(_DWORD *)(v7 + 36) == 120 )
      {
LABEL_4:
        if ( v3 == v4 )
        {
          v3 = *(_QWORD *)(v3 + 8);
        }
        else
        {
          v5 = v3 - 24;
          if ( !v3 )
            v5 = 0;
          sub_15F22F0((_QWORD *)(v4 - 24), v5);
        }
        goto LABEL_8;
      }
      v4 = *(_QWORD *)(v4 + 8);
    }
    while ( v2 != v4 );
  }
  return v3;
}
