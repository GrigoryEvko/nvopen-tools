// Function: sub_C089A0
// Address: 0xc089a0
//
__int64 __fastcall sub_C089A0(__int64 a1, __int64 a2)
{
  __int64 v2; // r12
  __int64 v3; // rbx
  char v4; // al
  char v5; // al
  char v7; // [rsp+Fh] [rbp-31h]

  v2 = a2 + 24;
  v3 = *(_QWORD *)(a2 + 32);
  v7 = 0;
  if ( v3 != a2 + 24 )
  {
    do
    {
      while ( 1 )
      {
        a2 = v3 - 56;
        if ( !v3 )
          a2 = 0;
        if ( sub_B2FC80(a2) )
          break;
        v3 = *(_QWORD *)(v3 + 8);
        if ( v2 == v3 )
          goto LABEL_8;
      }
      v4 = sub_C05FA0(*(_QWORD **)(a1 + 176), a2);
      v3 = *(_QWORD *)(v3 + 8);
      v7 |= v4 ^ 1;
    }
    while ( v2 != v3 );
  }
LABEL_8:
  v5 = sub_BF3D50(*(_QWORD *)(a1 + 176), a2);
  if ( *(_BYTE *)(a1 + 184) && (v5 != 1 || v7 || *(_BYTE *)(*(_QWORD *)(a1 + 176) + 153LL)) )
    sub_C64ED0("Broken module found, compilation aborted!", 1);
  return 0;
}
