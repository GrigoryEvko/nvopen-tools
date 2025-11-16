// Function: sub_309ED50
// Address: 0x309ed50
//
__int64 __fastcall sub_309ED50(int a1, const void **a2, __int64 a3, __int64 a4, __int64 *a5)
{
  __int64 v9; // [rsp+8h] [rbp-1C8h] BYREF
  __int64 v10[2]; // [rsp+10h] [rbp-1C0h] BYREF
  __int64 v11; // [rsp+20h] [rbp-1B0h]
  __int64 v12; // [rsp+28h] [rbp-1A8h]
  __int64 v13; // [rsp+30h] [rbp-1A0h]
  __int64 v14; // [rsp+38h] [rbp-198h]
  __int64 v15; // [rsp+40h] [rbp-190h]
  _BYTE v16[80]; // [rsp+50h] [rbp-180h] BYREF
  _BYTE v17[304]; // [rsp+A0h] [rbp-130h] BYREF

  sub_309D870(a1, a2, (__int64)v17);
  sub_CCBAC0((__int64)v16, (__int64)sub_309D7B0, (__int64)sub_309D7A0, 0, 0);
  sub_CD07E0((__int64)&v9, a3, (__int64)v17, (__int64)v16, 2);
  v15 = a4;
  v14 = 0x100000000LL;
  v10[0] = (__int64)&unk_49DD210;
  v10[1] = 0;
  v11 = 0;
  v12 = 0;
  v13 = 0;
  sub_CB5980((__int64)v10, 0, 0, 0);
  sub_CDD2D0(v9, (__int64)v10, 1, 1);
  if ( v13 != v11 )
    sub_CB5AE0(v10);
  v10[0] = (__int64)&unk_49DD210;
  sub_CB5840((__int64)v10);
  if ( v9 )
    (*(void (__fastcall **)(__int64))(*(_QWORD *)v9 + 56LL))(v9);
  sub_CEAF80(a5);
  return 1;
}
