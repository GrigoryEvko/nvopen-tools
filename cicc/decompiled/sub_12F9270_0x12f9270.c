// Function: sub_12F9270
// Address: 0x12f9270
//
__int64 __fastcall sub_12F9270(int a1, const void **a2, __int64 a3, __int64 a4, __int64 a5)
{
  __int64 v9; // [rsp+8h] [rbp-1B8h] BYREF
  void *v10; // [rsp+10h] [rbp-1B0h] BYREF
  __int64 v11; // [rsp+18h] [rbp-1A8h]
  __int64 v12; // [rsp+20h] [rbp-1A0h]
  __int64 v13; // [rsp+28h] [rbp-198h]
  int v14; // [rsp+30h] [rbp-190h]
  __int64 v15; // [rsp+38h] [rbp-188h]
  _BYTE v16[80]; // [rsp+40h] [rbp-180h] BYREF
  _BYTE v17[304]; // [rsp+90h] [rbp-130h] BYREF

  sub_12F7D90(a1, a2, (__int64)v17);
  sub_1C13840(v16, sub_12F7CD0, sub_12F7CC0, 0, 0);
  sub_1C17AF0(&v9, a3, v17, v16, 2);
  v14 = 1;
  v13 = 0;
  v10 = &unk_49EFBE0;
  v12 = 0;
  v11 = 0;
  v15 = a4;
  sub_1C23B90(v9, &v10, 1, 1);
  if ( v13 != v11 )
    sub_16E7BA0(&v10);
  sub_16E7BC0(&v10);
  if ( v9 )
    (*(void (__fastcall **)(__int64))(*(_QWORD *)v9 + 56LL))(v9);
  sub_1C3E9C0(a5);
  return 1;
}
