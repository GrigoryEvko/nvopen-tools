// Function: sub_8C84B0
// Address: 0x8c84b0
//
__int64 __fastcall sub_8C84B0(__int64 a1, __int64 a2)
{
  __int64 v2; // rbx
  __int64 v3; // rax
  __int64 *v4; // r15
  __int64 v5; // r13
  __int64 v6; // r14
  unsigned __int8 v8; // al
  __int64 v9; // rax
  __int64 *v10; // rax
  __int64 v11; // rax
  char i; // dl

  v2 = a2;
  v3 = sub_880F80(a1);
  v4 = *(__int64 **)(a1 + 88);
  if ( a2 )
  {
    v5 = v3;
    v6 = 0;
    while ( 1 )
    {
      if ( *(_DWORD *)(v2 + 40) == -1 || v5 == sub_880F80(v2) || !(unsigned int)sub_8C7F70(v2, a1) )
        goto LABEL_6;
      if ( !(unsigned int)sub_8C6B40(v2) )
      {
        v9 = sub_87D520(v2);
        if ( v9 && (*(_BYTE *)(v9 - 8) & 2) == 0 )
          *(_BYTE *)(v9 + 90) |= 8u;
        goto LABEL_6;
      }
      v8 = *(_BYTE *)(v2 + 80);
      if ( v8 == 7 )
      {
        if ( !v6 && *(__int64 **)(v2 + 88) != v4 )
          v6 = v2;
        goto LABEL_6;
      }
      if ( v8 > 7u )
        break;
      if ( v8 == 3 )
      {
        if ( *(_BYTE *)(v2 + 104) )
          goto LABEL_6;
        v11 = *(_QWORD *)(v2 + 88);
        for ( i = *(_BYTE *)(v11 + 140); i == 12; i = *(_BYTE *)(v11 + 140) )
          v11 = *(_QWORD *)(v11 + 160);
        if ( i == 14 )
          goto LABEL_6;
        goto LABEL_13;
      }
      if ( (unsigned __int8)(v8 - 4) > 2u )
        goto LABEL_13;
LABEL_6:
      v2 = *(_QWORD *)(v2 + 8);
      if ( !v2 )
        return v6;
    }
    if ( v8 == 14 )
    {
      if ( !v6 )
      {
        v10 = *(__int64 **)(*(_QWORD *)(v2 + 88) + 8LL);
        if ( v10 != v4 )
          v6 = *v10;
      }
      goto LABEL_6;
    }
LABEL_13:
    sub_8C6700(v4, (unsigned int *)(v2 + 48), 0x42Au, 0x425u);
    goto LABEL_6;
  }
  return 0;
}
