// Function: sub_5E4940
// Address: 0x5e4940
//
__int64 __fastcall sub_5E4940(__int64 a1)
{
  __int64 **v1; // rcx
  __int64 result; // rax
  __int64 i; // rax
  __int64 *v4; // rbx
  int v5; // r12d
  char v6; // al
  char v7; // dl

  v1 = *(__int64 ***)(a1 + 168);
  if ( (*(_BYTE *)(a1 + 176) & 0x50) != 0 || (result = 0, v1[10]) )
  {
    for ( i = *(_QWORD *)(a1 + 160); i; i = *(_QWORD *)(i + 112) )
    {
      if ( ((*(_BYTE *)(i + 144) & 4) == 0 || *(_BYTE *)(i + 137))
        && ((*(_BYTE *)(i + 144) & 0x40) == 0 || !*(_QWORD *)(i + 8)) )
      {
        return 0;
      }
    }
    v4 = *v1;
    if ( !*v1 )
      return 1;
    v5 = 0;
    while ( 1 )
    {
      while ( 1 )
      {
        v6 = *((_BYTE *)v4 + 96);
        if ( (v6 & 2) != 0 || (*(_BYTE *)(*(_QWORD *)(*(_QWORD *)(v4[14] + 8) + 16LL) + 96LL) & 2) != 0 )
          goto LABEL_13;
        v7 = *(_BYTE *)(v4[5] + 179) & 1;
        if ( (v6 & 1) == 0 )
          break;
        if ( v7 )
          goto LABEL_12;
        if ( !(unsigned int)sub_5E4940() || v5 )
          return 0;
        v5 = 1;
LABEL_13:
        v4 = (__int64 *)*v4;
        if ( !v4 )
          return 1;
      }
      if ( v7 )
      {
LABEL_12:
        if ( v4[13] )
          return 0;
        goto LABEL_13;
      }
      v4 = (__int64 *)*v4;
      if ( !v4 )
        return 1;
    }
  }
  return result;
}
