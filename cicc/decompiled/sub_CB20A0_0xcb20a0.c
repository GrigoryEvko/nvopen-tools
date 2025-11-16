// Function: sub_CB20A0
// Address: 0xcb20a0
//
char __fastcall sub_CB20A0(__int64 a1, char a2)
{
  __int64 v2; // rax
  unsigned int v3; // r14d
  __int64 v4; // r15
  __int64 v5; // r12
  unsigned int v6; // r13d
  unsigned int v7; // r12d
  int i; // r12d

  if ( *(_QWORD *)(a1 + 104) != 1 || **(_BYTE **)(a1 + 96) != 10 )
  {
    LOBYTE(v2) = (unsigned __int8)sub_CB1B10(a1, *(const void **)(a1 + 96), *(_QWORD *)(a1 + 104));
    *(_QWORD *)(a1 + 96) = 0;
    *(_QWORD *)(a1 + 104) = 0;
    return v2;
  }
  sub_CB1E40(a1);
  v2 = *(unsigned int *)(a1 + 40);
  *(_QWORD *)(a1 + 96) = 0;
  *(_QWORD *)(a1 + 104) = 0;
  v3 = v2;
  if ( v2 && !a2 )
  {
    v4 = *(_QWORD *)(a1 + 32);
    v5 = v4 + 4 * v2;
    LOBYTE(v2) = sub_CB2040(*(_DWORD *)(v5 - 4));
    if ( !(_BYTE)v2 )
    {
      --v3;
      LODWORD(v2) = *(_DWORD *)(v5 - 4) & 0xFFFFFFFD;
      if ( (_DWORD)v2 != 4 )
      {
        LOBYTE(v2) = sub_CB2090(*(_DWORD *)(v5 - 4));
        if ( !(_BYTE)v2 )
        {
          if ( !v3 )
            return v2;
          v6 = 0;
LABEL_11:
          v7 = v6;
          do
          {
            ++v7;
            LOBYTE(v2) = (unsigned __int8)sub_CB1B10(a1, "  ", 2u);
          }
          while ( v7 < v3 );
LABEL_13:
          if ( v6 )
          {
            for ( i = 0; i != v6; ++i )
              LOBYTE(v2) = (unsigned __int8)sub_CB1B10(a1, "- ", 2u);
          }
          return v2;
        }
      }
      v5 -= 4;
    }
    v6 = 0;
    do
    {
      if ( v4 == v5 )
        break;
      LOBYTE(v2) = sub_CB2040(*(_DWORD *)(v5 - 4));
      if ( !(_BYTE)v2 )
        break;
      LODWORD(v2) = *(_DWORD *)(v5 - 4);
      v5 -= 4;
      ++v6;
    }
    while ( !(_DWORD)v2 );
    if ( v6 >= v3 )
      goto LABEL_13;
    goto LABEL_11;
  }
  return v2;
}
