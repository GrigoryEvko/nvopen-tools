// Function: sub_166CFA0
// Address: 0x166cfa0
//
__int64 __fastcall sub_166CFA0(__int64 a1, __int64 a2)
{
  __int64 v2; // r12
  __int64 v3; // rbx
  __int64 v4; // rsi
  char v5; // al
  char v6; // al
  char v8; // [rsp+Fh] [rbp-31h]

  v2 = a2 + 24;
  v3 = *(_QWORD *)(a2 + 32);
  v8 = 0;
  if ( v3 != a2 + 24 )
  {
    do
    {
      while ( 1 )
      {
        v4 = v3 - 56;
        if ( !v3 )
          v4 = 0;
        if ( sub_15E4F60(v4) )
          break;
        v3 = *(_QWORD *)(v3 + 8);
        if ( v2 == v3 )
          goto LABEL_8;
      }
      v5 = sub_166A310(*(_QWORD **)(a1 + 160), v4);
      v3 = *(_QWORD *)(v3 + 8);
      v8 |= v5 ^ 1;
    }
    while ( v2 != v3 );
  }
LABEL_8:
  v6 = sub_165D700(*(_QWORD *)(a1 + 160));
  if ( *(_BYTE *)(a1 + 168) && (v6 != 1 || v8 || *(_BYTE *)(*(_QWORD *)(a1 + 160) + 73LL)) )
    sub_16BD130("Broken module found, compilation aborted!", 1);
  return 0;
}
