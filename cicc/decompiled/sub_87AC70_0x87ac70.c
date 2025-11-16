// Function: sub_87AC70
// Address: 0x87ac70
//
__int64 __fastcall sub_87AC70(__int64 a1, _DWORD *a2)
{
  __int64 v3; // rax
  __int64 v4; // rdx
  __int64 v5; // r12
  char v6; // al
  __int64 i; // rax
  _QWORD *v8; // rsi
  __int64 v9; // rax
  char j; // cl
  _BYTE v12[64]; // [rsp+0h] [rbp-40h] BYREF

  *a2 = 0;
  v3 = sub_82C1B0(a1, 0, 0, (__int64)v12);
  if ( v3 )
  {
    v4 = v3;
    v5 = 0;
    while ( 1 )
    {
      while ( 1 )
      {
        v6 = *(_BYTE *)(v4 + 80);
        if ( v6 == 16 )
        {
          if ( (*(_BYTE *)(v4 + 82) & 4) != 0 )
            goto LABEL_20;
          v4 = **(_QWORD **)(v4 + 88);
          v6 = *(_BYTE *)(v4 + 80);
          if ( v6 == 24 )
            break;
        }
        if ( (unsigned __int8)(v6 - 10) <= 1u )
          goto LABEL_10;
LABEL_4:
        if ( v6 == 17 )
          goto LABEL_10;
LABEL_5:
        v4 = sub_82C230(v12);
        if ( !v4 )
          return v5;
      }
      v4 = *(_QWORD *)(v4 + 88);
      v6 = *(_BYTE *)(v4 + 80);
      if ( (unsigned __int8)(v6 - 10) > 1u )
        goto LABEL_4;
LABEL_10:
      for ( i = *(_QWORD *)(*(_QWORD *)(v4 + 88) + 152LL); *(_BYTE *)(i + 140) == 12; i = *(_QWORD *)(i + 160) )
        ;
      v8 = **(_QWORD ***)(i + 168);
      v9 = v8[1];
      for ( j = *(_BYTE *)(v9 + 140); j == 12; j = *(_BYTE *)(v9 + 140) )
        v9 = *(_QWORD *)(v9 + 160);
      if ( !j || *v8 && (*(_BYTE *)(*v8 + 32LL) & 4) == 0 )
        goto LABEL_5;
      if ( v5 )
      {
LABEL_20:
        *a2 = 1;
        return 0;
      }
      v5 = v4;
      v4 = sub_82C230(v12);
      if ( !v4 )
        return v5;
    }
  }
  return 0;
}
