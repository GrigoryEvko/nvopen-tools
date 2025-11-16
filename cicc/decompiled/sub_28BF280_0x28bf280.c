// Function: sub_28BF280
// Address: 0x28bf280
//
__int64 __fastcall sub_28BF280(__int64 a1, __int64 *a2, __int64 *a3, __int64 a4, __int64 a5)
{
  char v8; // al
  unsigned __int8 v9; // dl
  __int64 v10; // r8
  __int64 v11; // r9
  int v12; // r13d
  int v14; // [rsp+0h] [rbp-90h] BYREF
  char *v15; // [rsp+8h] [rbp-88h]
  char v16; // [rsp+18h] [rbp-78h] BYREF
  char *v17; // [rsp+40h] [rbp-50h]
  char v18; // [rsp+50h] [rbp-40h] BYREF

  v8 = sub_B2D610(a1, 47);
  v9 = 1;
  if ( !v8 )
    v9 = sub_B2D610(a1, 18);
  sub_DFABC0((__int64)&v14, a3, v9, 1u);
  v12 = v14;
  if ( v17 != &v18 )
    _libc_free((unsigned __int64)v17);
  if ( v15 != &v16 )
    _libc_free((unsigned __int64)v15);
  if ( v12 )
    return sub_28BC410(a1, a2, a4, a5, v10, v11);
  else
    return 0;
}
