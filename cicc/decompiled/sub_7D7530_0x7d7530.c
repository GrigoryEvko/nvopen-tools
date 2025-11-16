// Function: sub_7D7530
// Address: 0x7d7530
//
__int64 __fastcall sub_7D7530(unsigned __int8 a1, const char *a2)
{
  __int64 v2; // r12
  size_t v3; // rax
  char *v4; // rax
  _QWORD *v5; // rax
  unsigned __int64 v6; // r13

  v2 = sub_7E16B0(10);
  v3 = strlen(a2);
  v4 = (char *)sub_7247C0(v3 + 1);
  *(_QWORD *)(v2 + 8) = v4;
  strcpy(v4, a2);
  v5 = sub_7259C0(8);
  v5[22] = 2;
  v6 = (unsigned __int64)v5;
  v5[20] = sub_72C610(a1);
  sub_8D6090(v6);
  sub_7E1B70("_Vals");
  sub_7E1C00(v2, v6);
  return v2;
}
