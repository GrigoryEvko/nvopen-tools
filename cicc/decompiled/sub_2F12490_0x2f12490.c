// Function: sub_2F12490
// Address: 0x2f12490
//
char __fastcall sub_2F12490(__int64 a1, _BYTE *a2, __int64 a3)
{
  __int64 v3; // rcx
  _BYTE *v5; // rax
  int v6; // esi
  _BYTE *v7; // rax
  unsigned __int8 *v8; // rax
  size_t v9; // rdx

  v3 = a3;
  if ( *a2 > 3u )
  {
    if ( *a2 <= 0x15u )
    {
      v7 = *(_BYTE **)(a1 + 32);
      if ( (unsigned __int64)v7 >= *(_QWORD *)(a1 + 24) )
      {
        sub_CB5D20(a1, 96);
        v3 = a3;
      }
      else
      {
        *(_QWORD *)(a1 + 32) = v7 + 1;
        *v7 = 96;
      }
      sub_A5C020(a2, a1, 1, v3);
      v5 = *(_BYTE **)(a1 + 32);
      if ( (unsigned __int64)v5 >= *(_QWORD *)(a1 + 24) )
      {
        LOBYTE(v5) = sub_CB5D20(a1, 96);
      }
      else
      {
        *(_QWORD *)(a1 + 32) = v5 + 1;
        *v5 = 96;
      }
    }
    else
    {
      sub_904010(a1, "%ir.");
      if ( (a2[7] & 0x10) != 0 )
      {
        v8 = (unsigned __int8 *)sub_BD5D20((__int64)a2);
        LOBYTE(v5) = (unsigned __int8)sub_A54F00(a1, v8, v9);
      }
      else
      {
        v6 = -1;
        if ( *(_QWORD *)(a3 + 32) )
          v6 = sub_A5A720(a3, (__int64)a2);
        LOBYTE(v5) = sub_2EAC190(a1, v6);
      }
    }
  }
  else
  {
    LOBYTE(v5) = sub_A5C020(a2, a1, 0, a3);
  }
  return (char)v5;
}
