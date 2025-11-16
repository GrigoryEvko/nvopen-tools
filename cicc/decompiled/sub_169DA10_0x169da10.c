// Function: sub_169DA10
// Address: 0x169da10
//
__int64 __fastcall sub_169DA10(__int16 **a1, __int64 a2)
{
  unsigned int v2; // r15d
  char v3; // al
  char v4; // dl
  bool v5; // r12
  char v6; // dl
  int v8; // r15d
  int v9; // r15d
  __int64 v10[2]; // [rsp+10h] [rbp-90h] BYREF
  char v11; // [rsp+22h] [rbp-7Eh]
  __int16 *v12[4]; // [rsp+30h] [rbp-70h] BYREF
  __int16 *v13[10]; // [rsp+50h] [rbp-50h] BYREF

  v2 = sub_1699430(a1, a2);
  v3 = *((_BYTE *)a1 + 18);
  v4 = v3 & 7;
  v5 = (v3 & 8) != 0;
  if ( (v3 & 6) == 0 )
    goto LABEL_7;
  if ( v4 != 3 )
  {
    while ( 1 )
    {
      v6 = *(_BYTE *)(a2 + 18);
      if ( (v6 & 6) == 0 || (v6 & 7) == 3 )
      {
        v4 = v3 & 7;
        goto LABEL_7;
      }
      if ( !(unsigned int)sub_1698CF0((__int64)a1, a2) )
        break;
      v8 = sub_169C2F0((__int64)a1);
      v9 = v8 - sub_169C2F0(a2);
      sub_16986C0(v13, (__int64 *)a2);
      sub_169C390((__int64)v10, v13, v9, 0);
      sub_1698460((__int64)v13);
      if ( !(unsigned int)sub_1698CF0((__int64)a1, (__int64)v10) )
      {
        sub_16986C0(v12, v10);
        sub_169C390((__int64)v13, v12, -1, 0);
        sub_16983E0((__int64)v10, (__int64)v13);
        sub_1698460((__int64)v13);
        sub_1698460((__int64)v12);
      }
      v11 = *((_BYTE *)a1 + 18) & 8 | v11 & 0xF7;
      v2 = sub_169D430(a1, v10, 0);
      sub_1698460((__int64)v10);
      v3 = *((_BYTE *)a1 + 18);
      v4 = v3 & 7;
      if ( (v3 & 6) == 0 )
        goto LABEL_7;
      if ( v4 == 3 )
        goto LABEL_3;
    }
    v4 = *((_BYTE *)a1 + 18) & 7;
LABEL_7:
    if ( v4 != 3 )
      return v2;
  }
LABEL_3:
  *((_BYTE *)a1 + 18) = *((_BYTE *)a1 + 18) & 0xF7 | (8 * v5);
  return v2;
}
