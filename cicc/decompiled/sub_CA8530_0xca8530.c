// Function: sub_CA8530
// Address: 0xca8530
//
__int64 __fastcall sub_CA8530(__int64 a1)
{
  char *v1; // rax
  char v2; // al
  unsigned int v4; // r12d

  v1 = *(char **)(a1 + 40);
  if ( v1 == *(char **)(a1 + 48) )
    return 0;
  v2 = *v1;
  if ( (unsigned __int8)(v2 - 49) > 8u )
    return 0;
  v4 = (char)(v2 - 48);
  sub_CA7F70(a1, 1u);
  return v4;
}
