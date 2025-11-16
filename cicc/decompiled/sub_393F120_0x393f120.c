// Function: sub_393F120
// Address: 0x393f120
//
bool __fastcall sub_393F120(__int64 a1)
{
  char *v1; // rax
  __int64 v2; // r8
  int v3; // ecx
  char v4; // dl
  __int64 v5; // rsi

  v1 = *(char **)(a1 + 8);
  v2 = 0;
  v3 = 0;
  v4 = *v1;
  v5 = *v1 & 0x7F;
  while ( 1 )
  {
    if ( (unsigned __int64)(v5 << v3) >> v3 != v5 )
      return 0;
    v2 += v5 << v3;
    v3 += 7;
    ++v1;
    if ( v4 >= 0 )
      break;
    v4 = *v1;
    v5 = *v1 & 0x7F;
    if ( v3 == 70 )
      return 0;
  }
  return v2 == 0x5350524F463432FFLL;
}
