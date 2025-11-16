// Function: sub_253C860
// Address: 0x253c860
//
__int64 *__fastcall sub_253C860(__int64 *a1, __int64 a2)
{
  char v2; // al
  const char *v3; // rsi
  bool v5; // zf

  v2 = *(_BYTE *)(a2 + 97);
  if ( (v2 & 3) == 3 )
  {
    v5 = *(_BYTE *)(a2 + 184) == 0;
    v3 = "assumed-dead-users";
    if ( !v5 )
      v3 = "assumed-dead";
  }
  else
  {
    v3 = "assumed-dead-users";
    if ( !v2 )
      v3 = "assumed-live";
  }
  sub_253C590(a1, v3);
  return a1;
}
