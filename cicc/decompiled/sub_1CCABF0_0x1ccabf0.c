// Function: sub_1CCABF0
// Address: 0x1ccabf0
//
__int64 __fastcall sub_1CCABF0(int a1, __int64 a2)
{
  size_t v2; // r12
  char *v3; // r13
  __int64 *v4; // rax
  __int64 result; // rax
  __int64 v6[5]; // [rsp+8h] [rbp-28h] BYREF

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
  v6[0] = *(_QWORD *)(a2 + 112);
  v4 = (__int64 *)sub_15E0530(a2);
  result = sub_1563170(v6, v4, -1, v3, v2);
  *(_QWORD *)(a2 + 112) = result;
  return result;
}
