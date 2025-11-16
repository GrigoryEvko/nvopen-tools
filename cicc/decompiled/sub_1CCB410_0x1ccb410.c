// Function: sub_1CCB410
// Address: 0x1ccb410
//
bool __fastcall sub_1CCB410(__int64 a1)
{
  __int64 v2; // rdi
  size_t v4; // rdx
  char *v5; // r13

  if ( *(_BYTE *)(a1 + 16) <= 0x17u )
    return 0;
  v2 = *(_QWORD *)(*(_QWORD *)(a1 + 40) + 56LL);
  if ( !v2 || !sub_1CCAC90(v2) )
    return 0;
  v4 = 0;
  v5 = off_4CD4978[0];
  if ( off_4CD4978[0] )
    v4 = strlen(off_4CD4978[0]);
  return (*(_QWORD *)(a1 + 48) || *(__int16 *)(a1 + 18) < 0) && sub_1625940(a1, v5, v4) != 0;
}
