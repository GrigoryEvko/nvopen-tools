// Function: sub_F06F10
// Address: 0xf06f10
//
char __fastcall sub_F06F10(__int64 a1)
{
  unsigned __int8 *v1; // rcx
  __int64 v2; // rsi
  unsigned __int8 v3; // al
  unsigned __int8 v4; // di
  char v5; // al
  __int64 v6; // rax
  __int64 v7; // rdx
  __int64 v8; // rax

  v1 = *(unsigned __int8 **)(a1 - 64);
  v2 = *(_QWORD *)(a1 - 32);
  v3 = *v1;
  v4 = *(_BYTE *)v2 - 42;
  if ( *v1 <= 0x1Cu || (unsigned int)v3 - 42 > 0x11 )
  {
    if ( v4 > 0x11u )
      goto LABEL_11;
  }
  else
  {
    v5 = (v3 - 46) & 0xF7;
    if ( v4 > 0x11u )
    {
      if ( !v5 )
      {
        v6 = *((_QWORD *)v1 + 2);
        if ( v6 )
          LOBYTE(v6) = *(_QWORD *)(v6 + 8) == 0;
        return v6;
      }
LABEL_11:
      LOBYTE(v6) = 0;
      return v6;
    }
    if ( !v5 )
    {
      v8 = *((_QWORD *)v1 + 2);
      if ( v8 )
      {
        if ( !*(_QWORD *)(v8 + 8) )
        {
          LOBYTE(v6) = 1;
          return v6;
        }
      }
    }
  }
  LOBYTE(v6) = 0;
  if ( ((*(_BYTE *)v2 - 46) & 0xF7) == 0 )
  {
    v7 = *(_QWORD *)(v2 + 16);
    if ( v7 )
      LOBYTE(v6) = *(_QWORD *)(v7 + 8) == 0;
  }
  return v6;
}
