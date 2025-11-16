// Function: sub_16C3AD0
// Address: 0x16c3ad0
//
__int64 __fastcall sub_16C3AD0(char *a1, unsigned __int64 a2, int a3)
{
  unsigned __int64 v3; // rax
  __int64 result; // rax
  char *v6; // rax
  bool v7; // r8
  bool v8; // zf
  const char *v9; // r12
  size_t v10; // rax
  char *v11; // [rsp+0h] [rbp-20h] BYREF
  unsigned __int64 v12; // [rsp+8h] [rbp-18h]

  v3 = a2;
  v11 = a1;
  v12 = a2;
  if ( a3 )
  {
LABEL_9:
    if ( v3 <= 3 )
    {
LABEL_3:
      if ( !v3 )
        return -1;
      goto LABEL_14;
    }
    v6 = v11;
    goto LABEL_11;
  }
  if ( a2 <= 2 )
    goto LABEL_3;
  v6 = a1;
  if ( a1[1] == 58 )
  {
    v7 = sub_16C36C0(a1[2], 0);
    result = 2;
    if ( v7 )
      return result;
    v3 = v12;
    goto LABEL_9;
  }
  if ( a2 == 3 )
    return -(__int64)!sub_16C36C0(*v6, a3);
LABEL_11:
  if ( !sub_16C36C0(*v6, a3) || v11[1] != *v11 || sub_16C36C0(v11[2], a3) )
  {
    if ( !v12 )
      return -1;
LABEL_14:
    v6 = v11;
    return -(__int64)!sub_16C36C0(*v6, a3);
  }
  v8 = a3 == 0;
  v9 = "/";
  if ( v8 )
    v9 = "\\/";
  v10 = strlen(v9);
  return sub_16D23E0(&v11, v9, v10, 2);
}
