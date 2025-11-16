// Function: sub_254F1D0
// Address: 0x254f1d0
//
_BOOL8 __fastcall sub_254F1D0(__int64 *a1, __int64 a2)
{
  char v2; // r8
  unsigned __int64 v3; // rax
  unsigned __int64 v4; // rdx
  char v6; // [rsp+Bh] [rbp-55h] BYREF
  int v7; // [rsp+Ch] [rbp-54h] BYREF
  __int128 v8; // [rsp+10h] [rbp-50h] BYREF
  __int128 v9; // [rsp+20h] [rbp-40h]
  _QWORD v10[6]; // [rsp+30h] [rbp-30h] BYREF

  v8 = 0;
  v9 = 0;
  v7 = sub_250CB50(a1 + 9, 0);
  v10[0] = &v7;
  v10[1] = a2;
  v10[2] = a1;
  v10[3] = &v8;
  v6 = 0;
  v2 = sub_2523890(a2, (__int64 (__fastcall *)(__int64, __int64 *))sub_2586F10, (__int64)v10, (__int64)a1, 1u, &v6);
  v3 = 1;
  if ( v2 )
  {
    v3 = 0x100000000LL;
    if ( BYTE8(v9) )
    {
      v3 = 1;
      if ( (_QWORD)v9 )
        v3 = v9;
      if ( v3 > 0x100000000LL )
        v3 = 0x100000000LL;
    }
  }
  v4 = a1[13];
  if ( v4 <= v3 )
    v3 = a1[13];
  if ( v3 < a1[12] )
    v3 = a1[12];
  a1[13] = v3;
  return v4 == v3;
}
