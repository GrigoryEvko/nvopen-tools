// Function: sub_163B6D0
// Address: 0x163b6d0
//
__int64 __fastcall sub_163B6D0(int *a1, __int64 *a2)
{
  char *v2; // r14
  size_t v3; // rdx
  __int64 v4; // rax
  __int64 v5; // rdx
  __int64 v6; // rax
  __int64 v7; // rdx
  __int64 v8; // rax
  __int64 v9; // rdx
  __int64 v10; // rax
  __int64 v11; // rdx
  __int64 v12; // rax
  __int64 v13; // rdx
  __int64 v14; // rax
  __int64 v15; // rdx
  __int64 v17; // [rsp+0h] [rbp-60h] BYREF
  __int64 v18; // [rsp+8h] [rbp-58h]
  __int64 v19; // [rsp+10h] [rbp-50h]
  __int64 v20; // [rsp+18h] [rbp-48h]
  __int64 v21; // [rsp+20h] [rbp-40h]
  __int64 v22; // [rsp+28h] [rbp-38h]
  __int64 v23; // [rsp+30h] [rbp-30h]
  __int64 v24; // [rsp+38h] [rbp-28h]

  v2 = (char *)*(&off_4CD28C0 + *a1);
  v17 = sub_161FF10(a2, "ProfileFormat", 0xDu);
  if ( v2 )
  {
    v3 = strlen(v2);
  }
  else
  {
    v2 = 0;
    v3 = 0;
  }
  v18 = sub_161FF10(a2, v2, v3);
  v4 = sub_1627350(a2, &v17, (__int64 *)2, 0, 1);
  v5 = *((_QWORD *)a1 + 4);
  v17 = v4;
  v6 = sub_163AE50(a2, "TotalCount", v5);
  v7 = *((_QWORD *)a1 + 5);
  v18 = v6;
  v8 = sub_163AE50(a2, "MaxCount", v7);
  v9 = *((_QWORD *)a1 + 6);
  v19 = v8;
  v10 = sub_163AE50(a2, "MaxInternalCount", v9);
  v11 = *((_QWORD *)a1 + 7);
  v20 = v10;
  v12 = sub_163AE50(a2, "MaxFunctionCount", v11);
  v13 = (unsigned int)a1[16];
  v21 = v12;
  v14 = sub_163AE50(a2, "NumCounts", v13);
  v15 = (unsigned int)a1[17];
  v22 = v14;
  v23 = sub_163AE50(a2, "NumFunctions", v15);
  v24 = sub_163B520((__int64)a1, a2);
  return sub_1627350(a2, &v17, (__int64 *)8, 0, 1);
}
