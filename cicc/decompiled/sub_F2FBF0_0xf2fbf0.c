// Function: sub_F2FBF0
// Address: 0xf2fbf0
//
char __fastcall sub_F2FBF0(__int64 **a1, __int64 a2)
{
  __int64 v2; // rbx
  __int64 v3; // rax
  __int64 v4; // r14
  signed __int64 v5; // rax
  __int64 v6; // r13
  __int64 v7; // r15
  const char *v8; // rax
  const char *v9; // rax
  const char *v10; // rax
  const char *v11; // rax
  char result; // al
  const char *v13; // rax
  const char *v14; // rax
  const char *v15; // rax

  v2 = **a1;
  v3 = 72LL * *((unsigned int *)*a1 + 2);
  v4 = v2 + v3;
  v5 = 0x8E38E38E38E38E39LL * (v3 >> 3);
  if ( v5 >> 2 )
  {
    v6 = v2 + 288 * (v5 >> 2);
    while ( 1 )
    {
      v11 = sub_BD5D20(a2);
      if ( (unsigned __int8)sub_1099960(v2, v11) )
        return v4 != v2;
      v7 = v2 + 72;
      v8 = sub_BD5D20(a2);
      if ( (unsigned __int8)sub_1099960(v2 + 72, v8) )
        return v4 != v7;
      v7 = v2 + 144;
      v9 = sub_BD5D20(a2);
      if ( (unsigned __int8)sub_1099960(v2 + 144, v9) )
        return v4 != v7;
      v7 = v2 + 216;
      v10 = sub_BD5D20(a2);
      if ( (unsigned __int8)sub_1099960(v2 + 216, v10) )
        return v4 != v7;
      v2 += 288;
      if ( v6 == v2 )
      {
        v5 = 0x8E38E38E38E38E39LL * ((v4 - v2) >> 3);
        break;
      }
    }
  }
  if ( v5 == 2 )
    goto LABEL_18;
  if ( v5 == 3 )
  {
    v13 = sub_BD5D20(a2);
    if ( (unsigned __int8)sub_1099960(v2, v13) )
      return v4 != v2;
    v2 += 72;
LABEL_18:
    v14 = sub_BD5D20(a2);
    if ( !(unsigned __int8)sub_1099960(v2, v14) )
    {
      v2 += 72;
      goto LABEL_20;
    }
    return v4 != v2;
  }
  if ( v5 != 1 )
    return 0;
LABEL_20:
  v15 = sub_BD5D20(a2);
  result = sub_1099960(v2, v15);
  if ( result )
    return v4 != v2;
  return result;
}
