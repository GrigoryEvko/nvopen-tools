// Function: sub_82B170
// Address: 0x82b170
//
__int64 __fastcall sub_82B170(__int64 a1, unsigned __int8 a2, _QWORD *a3)
{
  __int64 *v3; // r15
  char v5; // r12
  __int64 v6; // rbx
  __int64 v7; // rax
  char v8; // dl
  char *src; // [rsp+0h] [rbp-40h]
  char v11; // [rsp+Fh] [rbp-31h]

  v3 = (__int64 *)a1;
  src = (char *)qword_4F064C0[a2];
  sub_7461E0((__int64)&qword_4F5F780);
  qword_4F5F780 = (__int64)sub_729610;
  byte_4F5F811 = dword_4F07460;
  qword_4F06C40 = 0;
  if ( (unsigned __int8)(a2 - 1) > 3u && (unsigned __int8)(a2 - 41) > 1u )
  {
    v11 = 0;
    if ( !*(_QWORD *)a1 )
    {
      sub_7295A0(src);
      sub_729660(32);
      v11 = 1;
    }
    v5 = 0;
  }
  else
  {
    if ( !a1 )
      goto LABEL_17;
    v11 = 0;
    v5 = 1;
  }
  v6 = 1;
  if ( *(_BYTE *)(a1 + 8) )
    goto LABEL_12;
LABEL_6:
  sub_74B930(*(_QWORD *)(v3[3] + 8), (__int64)&qword_4F5F780);
  if ( v5 )
  {
LABEL_7:
    if ( !*v3 )
      goto LABEL_17;
    sub_7295A0(", ");
  }
  else
  {
    while ( v6 != 1 )
    {
      if ( v6 != 2 || a2 != 43 )
        goto LABEL_9;
      sub_7295A0(" ]");
      v7 = *v3;
      if ( !*v3 )
        goto LABEL_17;
LABEL_10:
      v8 = *(_BYTE *)(v7 + 8);
      ++v6;
      if ( v8 == 3 )
      {
        v7 = sub_6BBB10(v3);
        if ( !v7 )
          goto LABEL_17;
        v8 = *(_BYTE *)(v7 + 8);
      }
      v3 = (__int64 *)v7;
      if ( !v8 )
        goto LABEL_6;
LABEL_12:
      sub_7295A0("{...}");
      if ( v5 )
        goto LABEL_7;
    }
    if ( a2 == 43 )
    {
      sub_7295A0(" [ ");
    }
    else if ( a2 == 44 )
    {
      sub_7295A0(" : ");
    }
    else if ( !v11 )
    {
      sub_729660(32);
      sub_7295A0(src);
      if ( (unsigned __int8)(a2 - 37) <= 1u )
        goto LABEL_17;
      sub_729660(32);
    }
  }
LABEL_9:
  v7 = *v3;
  if ( *v3 )
    goto LABEL_10;
LABEL_17:
  sub_729660(0);
  return sub_67DCF0(a3, 740, (__int64)qword_4F06C50);
}
