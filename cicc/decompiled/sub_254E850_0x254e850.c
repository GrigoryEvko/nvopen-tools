// Function: sub_254E850
// Address: 0x254e850
//
__int64 *__fastcall sub_254E850(__int64 *a1, __int64 a2)
{
  unsigned __int8 v2; // al
  char v3; // al
  const char *v4; // rsi

  v2 = *(_BYTE *)sub_250D070((_QWORD *)(a2 + 72));
  if ( v2 > 0x1Cu )
  {
    if ( v2 == 62 )
    {
      if ( *(_BYTE *)(a2 + 97) )
      {
        sub_253C590(a1, "assumed-dead-store");
        return a1;
      }
    }
    else if ( v2 == 64 && *(_BYTE *)(a2 + 97) )
    {
      sub_253C590(a1, "assumed-dead-fence");
      return a1;
    }
  }
  v3 = (*(__int64 (__fastcall **)(__int64))(*(_QWORD *)a2 + 112LL))(a2);
  v4 = "assumed-dead";
  if ( !v3 )
    v4 = "assumed-live";
  sub_253C590(a1, v4);
  return a1;
}
