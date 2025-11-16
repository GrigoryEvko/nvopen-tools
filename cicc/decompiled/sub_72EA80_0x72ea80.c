// Function: sub_72EA80
// Address: 0x72ea80
//
__int64 __fastcall sub_72EA80(__int64 a1, _QWORD *a2, int a3)
{
  __int64 v5; // rbx
  char v6; // dl
  __int64 v8; // r12
  __int64 i; // r14
  __int64 j; // rbx
  __int64 v11; // rcx
  __int64 v12; // r8
  __int64 v13; // rax
  __int64 v14; // rcx
  __int64 v15; // r8
  __int64 v16; // rsi
  int v17; // [rsp+4h] [rbp-2Ch] BYREF
  _QWORD v18[5]; // [rsp+8h] [rbp-28h] BYREF

  v5 = a1;
  v6 = *(_BYTE *)(a1 + 173);
  if ( v6 == 12 )
  {
    while ( *(_BYTE *)(v5 + 176) == 1 )
    {
      if ( !(unsigned int)sub_72E9D0((_BYTE *)v5, v18, &v17) || *(_QWORD *)(v5 + 128) != *(_QWORD *)(v18[0] + 128LL) )
      {
        v6 = *(_BYTE *)(v5 + 173);
        goto LABEL_9;
      }
      v6 = *(_BYTE *)(v18[0] + 173LL);
      v5 = v18[0];
      if ( v6 != 12 )
        goto LABEL_9;
    }
    return 0;
  }
LABEL_9:
  if ( v6 != 6 || *(_BYTE *)(v5 + 176) != 1 || *(_QWORD *)(v5 + 192) )
    return 0;
  v8 = *(_QWORD *)(v5 + 184);
  if ( (*(_BYTE *)(v5 + 168) & 8) != 0 )
  {
    if ( !(unsigned int)sub_8D2E30(*(_QWORD *)(v5 + 128))
      || (v13 = sub_8D46C0(*(_QWORD *)(v5 + 128)), v16 = *(_QWORD *)(v8 + 120), v16 != v13)
      && !(unsigned int)sub_8D97D0(v13, v16, 0, v14, v15) )
    {
      if ( !a3 || !(unsigned int)sub_8D2E30(*(_QWORD *)(v5 + 128)) || !(unsigned int)sub_8D3410(*(_QWORD *)(v8 + 120)) )
        return 0;
      for ( i = sub_8D40F0(*(_QWORD *)(v8 + 120)); *(_BYTE *)(i + 140) == 12; i = *(_QWORD *)(i + 160) )
        ;
      for ( j = sub_8D46C0(*(_QWORD *)(v5 + 128)); *(_BYTE *)(j + 140) == 12; j = *(_QWORD *)(j + 160) )
        ;
      if ( (unsigned int)sub_8D3410(j) )
      {
        for ( j = sub_8D40F0(j); *(_BYTE *)(j + 140) == 12; j = *(_QWORD *)(j + 160) )
          ;
      }
      if ( j != i && !(unsigned int)sub_8D97D0(i, j, 0, v11, v12) )
        return 0;
    }
  }
  *a2 = v8;
  return 1;
}
