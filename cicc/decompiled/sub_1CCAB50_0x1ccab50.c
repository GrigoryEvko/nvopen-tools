// Function: sub_1CCAB50
// Address: 0x1ccab50
//
__int64 __fastcall sub_1CCAB50(int a1, __int64 a2)
{
  size_t v2; // r13
  char *v3; // r14
  __int64 *v4; // rax
  _QWORD *v5; // rax

  if ( a1 == 2 )
  {
    v2 = 0;
    v3 = off_4CD4960[0];
    if ( !off_4CD4960[0] )
      goto LABEL_5;
    goto LABEL_4;
  }
  if ( a1 != 3 )
  {
    v2 = 0;
    v3 = off_4CD4968[0];
    if ( !off_4CD4968[0] )
      goto LABEL_5;
    goto LABEL_4;
  }
  v2 = 0;
  v3 = off_4CD4958[0];
  if ( off_4CD4958[0] )
LABEL_4:
    v2 = strlen(v3);
LABEL_5:
  v4 = (__int64 *)sub_15E0530(a2);
  v5 = sub_155D020(v4, v3, v2, 0, 0);
  return sub_15E0DA0(a2, -1, (__int64)v5);
}
