// Function: sub_23B2150
// Address: 0x23b2150
//
__int64 __fastcall sub_23B2150(__int64 a1, unsigned __int64 a2)
{
  const char *v3; // rsi
  __int64 v4; // rsi
  __int64 v5; // rax
  __int64 v6; // rsi
  unsigned int v7; // edx
  unsigned __int64 v8; // r13
  __int64 v9; // rax
  __int64 v10; // rax
  __int64 v11; // rax
  __int64 v12; // rax
  const char *v14; // rax
  size_t v15; // rdx

  if ( (*(_BYTE *)(a2 + 7) & 0x10) != 0 )
  {
    v14 = sub_BD5D20(a2);
    v10 = sub_A51340(a1, v14, v15);
LABEL_11:
    a1 = v10;
    v3 = "<";
    goto LABEL_12;
  }
  v3 = "unnamed_removed<";
  if ( *(_QWORD *)(a2 + 72) )
  {
    if ( sub_AA5B70(a2) )
    {
      v10 = sub_904010(a1, "entry");
    }
    else
    {
      v4 = *(_QWORD *)(a2 + 72);
      v5 = *(_QWORD *)(v4 + 80);
      v6 = v4 + 72;
      if ( v6 == v5 )
      {
        v8 = 0;
      }
      else
      {
        v7 = 0;
        do
        {
          if ( v5 && a2 == v5 - 24 )
            break;
          v5 = *(_QWORD *)(v5 + 8);
          ++v7;
        }
        while ( v6 != v5 );
        v8 = v7;
      }
      v9 = sub_904010(a1, "unnamed_");
      v10 = sub_CB59D0(v9, v8);
    }
    goto LABEL_11;
  }
LABEL_12:
  v11 = sub_904010(a1, v3);
  v12 = sub_CB5A80(v11, a2);
  return sub_904010(v12, ">");
}
