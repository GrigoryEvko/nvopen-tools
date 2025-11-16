// Function: sub_EED260
// Address: 0xeed260
//
__int64 __fastcall sub_EED260(char **a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  __int64 v6; // rbp
  char *v7; // rax
  char v8; // al
  __int64 result; // rax
  __int64 v10; // rdx
  __int64 v11; // rcx
  __int64 v12; // r8
  __int64 v13; // r9
  char **v14; // [rsp-20h] [rbp-20h]
  __int64 v15[2]; // [rsp-10h] [rbp-10h] BYREF

  v7 = *a1;
  if ( a1[1] == *a1 )
    return sub_EE9AE0((__int64)a1, a2, a3, a4, a5, a6);
  v15[1] = v6;
  v8 = *v7;
  if ( v8 == 84 )
  {
    v14 = a1;
    result = sub_EEA020((__int64)a1, a2, a3, a4, a5);
    v15[0] = result;
    if ( !result )
      return result;
    goto LABEL_7;
  }
  if ( v8 != 68 )
    return sub_EE9AE0((__int64)a1, a2, a3, a4, a5, a6);
  v14 = a1;
  result = sub_EED1D0(a1);
  v15[0] = result;
  if ( result )
  {
LABEL_7:
    sub_E18380((__int64)(v14 + 37), v15, v10, v11, v12, v13);
    return v15[0];
  }
  return result;
}
