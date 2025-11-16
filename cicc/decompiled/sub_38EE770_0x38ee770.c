// Function: sub_38EE770
// Address: 0x38ee770
//
__int64 __fastcall sub_38EE770(_QWORD *a1, __int64 a2, __int64 a3)
{
  __int64 v3; // r8
  __int64 v4; // r9
  unsigned int v5; // r13d
  __int64 v6; // rsi
  __int64 i; // rax
  __int64 v8; // rdx
  _QWORD v10[2]; // [rsp+0h] [rbp-80h] BYREF
  const char *v11; // [rsp+10h] [rbp-70h] BYREF
  _QWORD *v12; // [rsp+18h] [rbp-68h]
  __int16 v13; // [rsp+20h] [rbp-60h]
  const char **v14; // [rsp+30h] [rbp-50h] BYREF
  char *v15; // [rsp+38h] [rbp-48h]
  __int16 v16; // [rsp+40h] [rbp-40h]

  v10[0] = a2;
  v10[1] = a3;
  v11 = "unexpected token in '";
  v16 = 770;
  v13 = 1283;
  v12 = v10;
  v14 = &v11;
  v15 = "' directive";
  v5 = sub_3909E20(a1, 9, &v14);
  if ( !(_BYTE)v5 )
  {
    v6 = a1[57];
    if ( a1[56] == v6 )
    {
      v13 = 1283;
      v11 = "unexpected '";
      v16 = 770;
      v12 = v10;
      v14 = &v11;
      v15 = "' in file, no current macro definition";
      return (unsigned int)sub_3909CF0(a1, &v14, 0, 0, v3, v4);
    }
    else
    {
      for ( i = a1[50]; *(_QWORD *)(*(_QWORD *)(v6 - 8) + 24LL) != (i - a1[49]) >> 3; *(_QWORD *)((char *)a1 + 380) = v8 )
      {
        v8 = *(_QWORD *)(i - 8);
        i -= 8;
        a1[50] = i;
      }
      sub_38EE710((__int64)a1);
    }
  }
  return v5;
}
