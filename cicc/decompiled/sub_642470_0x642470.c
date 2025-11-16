// Function: sub_642470
// Address: 0x642470
//
__int64 __fastcall sub_642470(__int64 a1, __int64 a2)
{
  __int64 v2; // rax
  __int64 i; // rbx
  __int64 *v4; // rdx
  __int64 v5; // r14
  unsigned int v6; // r14d
  __int64 *v7; // rbx
  __int64 v8; // rcx
  char v9; // al
  char v11; // dl

  v2 = *(_QWORD *)(a1 + 176);
  for ( i = *(_QWORD *)(v2 + 152); *(_BYTE *)(i + 140) == 12; i = *(_QWORD *)(i + 160) )
    ;
  v4 = *(__int64 **)(i + 168);
  v5 = *v4;
  if ( (*(_BYTE *)(v2 + 88) & 0x70) == 0x30 )
  {
    if ( a2 )
    {
      sub_6851C0(2489, a2);
      if ( v5 || (*(_BYTE *)(*(_QWORD *)(i + 168) + 16LL) & 1) != 0 )
        goto LABEL_6;
    }
    goto LABEL_14;
  }
  if ( v5 || (v6 = 1, (v4[2] & 1) != 0) )
  {
    if ( a2 )
    {
LABEL_6:
      v6 = 0;
      sub_6851C0(2499, a2);
      goto LABEL_7;
    }
LABEL_14:
    v6 = 0;
  }
LABEL_7:
  v7 = **(__int64 ***)(a1 + 328);
  v8 = *v7;
  v9 = *(_BYTE *)(v7[1] + 80);
  if ( v9 != 3 || !unk_4F07700 )
  {
    if ( dword_4F077C4 == 2 && unk_4F07778 > 202001 )
    {
      if ( v8 )
        goto LABEL_10;
      v11 = v7[7] & 0x10;
      if ( v9 != 2 )
      {
LABEL_29:
        if ( !v11 )
          goto LABEL_10;
LABEL_30:
        if ( *(_BYTE *)(v7[1] + 80) == 2 && (unsigned int)sub_8D2A50(*(_QWORD *)(v7[8] + 128)) )
          return v6;
        goto LABEL_10;
      }
      if ( v11 )
        goto LABEL_30;
      if ( (unsigned int)sub_8D3A70(*(_QWORD *)(v7[8] + 128)) || (unsigned int)sub_8D3F60(*(_QWORD *)(v7[8] + 128)) )
        return v6;
    }
    if ( *v7 )
    {
LABEL_10:
      if ( !a2 )
        return 0;
      v6 = 0;
      sub_6851C0(2500, a2);
      return v6;
    }
    v11 = v7[7] & 0x10;
    goto LABEL_29;
  }
  if ( v8
    && !*(_QWORD *)v8
    && (*(_BYTE *)(v8 + 56) & 0x10) != 0
    && *(_BYTE *)(*(_QWORD *)(v8 + 8) + 80LL) == 2
    && v7[8] == *(_QWORD *)(*(_QWORD *)(v8 + 64) + 128LL) )
  {
    return v6;
  }
  if ( a2 )
    sub_6851C0(2905, a2);
  return 0;
}
