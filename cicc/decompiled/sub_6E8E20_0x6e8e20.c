// Function: sub_6E8E20
// Address: 0x6e8e20
//
__int64 __fastcall sub_6E8E20(__int64 a1)
{
  __int64 v1; // r12
  char v2; // al
  __int64 v3; // rax
  char v4; // dl
  __int64 i; // rax
  unsigned __int8 v6; // bl
  __int64 v7; // r12
  __int64 v8; // rax

  v1 = a1;
  v2 = *(_BYTE *)(a1 + 140);
  if ( v2 == 12 )
  {
    v3 = a1;
    do
    {
      v3 = *(_QWORD *)(v3 + 160);
      v4 = *(_BYTE *)(v3 + 140);
    }
    while ( v4 == 12 );
    if ( !v4 )
      return sub_72C930(a1);
    do
      v1 = *(_QWORD *)(v1 + 160);
    while ( *(_BYTE *)(v1 + 140) == 12 );
  }
  else if ( !v2 )
  {
    return sub_72C930(a1);
  }
  for ( i = *(_QWORD *)(v1 + 160); *(_BYTE *)(i + 140) == 12; i = *(_QWORD *)(i + 160) )
    ;
  v6 = sub_622A10(*(_QWORD *)(i + 128), *(_DWORD *)(i + 136), 1);
  if ( v6 != 13 )
  {
    v7 = sub_8D4620(v1);
    v8 = sub_72BA30(v6);
    return sub_72B5A0(v8, v7, 0);
  }
  a1 = 3261;
  sub_685360(0xCBDu, dword_4F07508, v1);
  return sub_72C930(a1);
}
