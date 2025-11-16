// Function: sub_C80650
// Address: 0xc80650
//
__int64 __fastcall sub_C80650(char *a1, unsigned __int64 a2, unsigned int a3)
{
  unsigned __int64 v3; // rax
  __int64 result; // rax
  char *v6; // rax
  bool v7; // cf
  const char *v8; // r12
  size_t v9; // rax
  bool v10; // r8
  char *v11; // [rsp+0h] [rbp-20h] BYREF
  unsigned __int64 v12; // [rsp+8h] [rbp-18h]

  v3 = a2;
  v11 = a1;
  v12 = a2;
  if ( a3 <= 1 )
  {
LABEL_2:
    if ( v3 <= 3 )
      goto LABEL_3;
    v6 = v11;
    goto LABEL_7;
  }
  if ( a2 <= 2 )
  {
LABEL_3:
    if ( !v3 )
      return -1;
    goto LABEL_10;
  }
  v6 = a1;
  if ( a1[1] == 58 )
  {
    v10 = sub_C80220(a1[2], a3);
    result = 2;
    if ( v10 )
      return result;
    v3 = v12;
    goto LABEL_2;
  }
  if ( a2 == 3 )
    return -(__int64)!sub_C80220(*v6, a3);
LABEL_7:
  if ( !sub_C80220(*v6, a3) || v11[1] != *v11 || sub_C80220(v11[2], a3) )
  {
    if ( !v12 )
      return -1;
LABEL_10:
    v6 = v11;
    return -(__int64)!sub_C80220(*v6, a3);
  }
  v7 = a3 < 2;
  v8 = "/";
  if ( !v7 )
    v8 = "\\/";
  v9 = strlen(v8);
  return sub_C934D0(&v11, v8, v9, 2);
}
