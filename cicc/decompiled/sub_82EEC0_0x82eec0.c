// Function: sub_82EEC0
// Address: 0x82eec0
//
_BOOL8 __fastcall sub_82EEC0(__int64 a1)
{
  unsigned int v1; // r12d
  int v2; // r13d
  __int64 v3; // rax
  int v4; // r12d
  char v5; // dl
  __int64 i; // rax
  __int64 *v7; // rax
  _BOOL8 result; // rax
  _BYTE v9[64]; // [rsp+0h] [rbp-40h] BYREF

  v1 = dword_4F077BC;
  v2 = qword_4F077B4;
  if ( ((unsigned int)sub_880A60()
     || (*(_BYTE *)(a1 + 81) & 0x10) != 0 && (*(_BYTE *)(*(_QWORD *)(a1 + 64) + 177LL) & 0xA0) != 0)
    && (v3 = sub_82C1B0(a1, 0, 0, (__int64)v9)) != 0 )
  {
    v4 = v2 | v1;
    while ( 1 )
    {
      v5 = *(_BYTE *)(v3 + 80);
      if ( v5 == 16 )
      {
        v3 = **(_QWORD **)(v3 + 88);
        v5 = *(_BYTE *)(v3 + 80);
      }
      if ( v5 == 24 )
      {
        v3 = *(_QWORD *)(v3 + 88);
        v5 = *(_BYTE *)(v3 + 80);
      }
      if ( v5 == 20 || v5 == 2 )
        return 1;
      for ( i = *(_QWORD *)(*(_QWORD *)(v3 + 88) + 152LL); *(_BYTE *)(i + 140) == 12; i = *(_QWORD *)(i + 160) )
        ;
      v7 = *(__int64 **)(i + 168);
      if ( v4 )
      {
        if ( (*((_BYTE *)v7 + 21) & 1) != 0 )
          return 1;
      }
      while ( 1 )
      {
        v7 = (__int64 *)*v7;
        if ( !v7 )
          break;
        if ( (v7[4] & 0x180) != 0 || (v7[4] & 4) != 0 )
          return 1;
      }
      v3 = sub_82C230(v9);
      if ( !v3 )
        goto LABEL_21;
    }
  }
  else
  {
LABEL_21:
    result = 1;
    if ( (*(_BYTE *)(a1 + 84) & 4) == 0 )
      return (unsigned int)sub_82EE30() != 0;
  }
  return result;
}
