// Function: sub_1CCAAC0
// Address: 0x1ccaac0
//
bool __fastcall sub_1CCAAC0(int a1, __int64 a2)
{
  size_t v2; // rdx
  char *v3; // r12

  if ( a1 == 2 )
  {
    v3 = off_4CD4960[0];
    if ( off_4CD4960[0] )
      goto LABEL_4;
  }
  else
  {
    if ( a1 == 3 )
    {
      v2 = 0;
      v3 = off_4CD4958[0];
      if ( !off_4CD4958[0] )
        return sub_15602E0((_QWORD *)(a2 + 112), v3, v2);
LABEL_4:
      v2 = strlen(v3);
      return sub_15602E0((_QWORD *)(a2 + 112), v3, v2);
    }
    v3 = off_4CD4968[0];
    if ( off_4CD4968[0] )
      goto LABEL_4;
  }
  return sub_15602E0((_QWORD *)(a2 + 112), 0, 0);
}
