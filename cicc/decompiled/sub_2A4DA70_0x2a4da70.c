// Function: sub_2A4DA70
// Address: 0x2a4da70
//
void __fastcall sub_2A4DA70(__int64 a1, char a2)
{
  __int64 v2; // r13
  __int64 v3; // r12
  _QWORD *v4; // r15
  __int64 v5; // rbx
  __int64 v6; // r14
  _QWORD *v7; // r12
  __int64 v8; // rax
  __int64 v9; // rax

  v2 = *(_QWORD *)(a1 + 16);
  while ( v2 )
  {
    while ( 1 )
    {
      v3 = v2;
      v2 = *(_QWORD *)(v2 + 8);
      v4 = *(_QWORD **)(v3 + 24);
      if ( (unsigned __int8)(*(_BYTE *)v4 - 61) > 1u )
        break;
LABEL_5:
      if ( !v2 )
        return;
    }
    if ( sub_BD2BE0(*(_QWORD *)(v3 + 24)) )
    {
      sub_BD5D50(v3);
      goto LABEL_5;
    }
    if ( *(_BYTE *)(v4[1] + 8LL) != 7 )
    {
      v5 = v4[2];
      while ( v5 )
      {
        v6 = v5;
        v5 = *(_QWORD *)(v5 + 8);
        v7 = *(_QWORD **)(v6 + 24);
        if ( a2 )
        {
          if ( *(_BYTE *)v7 != 85 )
            continue;
          v9 = *(v7 - 4);
          if ( !v9 || *(_BYTE *)v9 || *(_QWORD *)(v9 + 24) != v7[10] || (*(_BYTE *)(v9 + 33) & 0x20) == 0 )
            continue;
        }
        if ( sub_BD2BE0(*(_QWORD *)(v6 + 24)) )
          sub_BD5D50(v6);
        else
          sub_B43D60(v7);
      }
    }
    if ( a2 )
    {
      if ( *(_BYTE *)v4 != 85 )
        goto LABEL_5;
      v8 = *(v4 - 4);
      if ( !v8 || *(_BYTE *)v8 || *(_QWORD *)(v8 + 24) != v4[10] || (*(_BYTE *)(v8 + 33) & 0x20) == 0 )
        goto LABEL_5;
    }
    sub_B43D60(v4);
  }
}
