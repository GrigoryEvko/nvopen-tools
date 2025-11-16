// Function: sub_620E90
// Address: 0x620e90
//
__int64 __fastcall sub_620E90(__int64 a1)
{
  __int64 v1; // rax
  char i; // dl
  unsigned int v3; // r8d

  v1 = *(_QWORD *)(a1 + 128);
  for ( i = *(_BYTE *)(v1 + 140); i == 12; i = *(_BYTE *)(v1 + 140) )
    v1 = *(_QWORD *)(v1 + 160);
  v3 = 0;
  if ( i == 2 )
    return byte_4B6DF90[*(unsigned __int8 *)(v1 + 160)] != 0;
  return v3;
}
