// Function: sub_38EE8B0
// Address: 0x38ee8b0
//
__int64 __fastcall sub_38EE8B0(_QWORD *a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  _DWORD *v6; // rax
  char *v8; // rax
  _QWORD v9[2]; // [rsp+0h] [rbp-50h] BYREF
  const char *v10; // [rsp+10h] [rbp-40h] BYREF
  _QWORD *v11; // [rsp+18h] [rbp-38h]
  __int16 v12; // [rsp+20h] [rbp-30h]
  _QWORD v13[2]; // [rsp+30h] [rbp-20h] BYREF
  __int16 v14; // [rsp+40h] [rbp-10h]

  v6 = (_DWORD *)a1[19];
  v9[0] = a2;
  v9[1] = a3;
  if ( *v6 == 9 )
  {
    if ( a1[57] != a1[56] )
    {
      sub_38EE710((__int64)a1);
      return 0;
    }
    v10 = "unexpected '";
    v12 = 1283;
    v11 = v9;
    v13[0] = &v10;
    v8 = "' in file, no current macro definition";
  }
  else
  {
    v10 = "unexpected token in '";
    v11 = v9;
    v13[0] = &v10;
    v8 = "' directive";
    v12 = 1283;
  }
  v13[1] = v8;
  v14 = 770;
  return sub_3909CF0(a1, v13, 0, 0, a5, a6);
}
