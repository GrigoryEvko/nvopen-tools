// Function: sub_21141B0
// Address: 0x21141b0
//
__int64 __fastcall sub_21141B0(_QWORD *a1, __int64 *a2, __int64 a3, _BYTE *a4, int a5, _BYTE *a6)
{
  __int64 v9; // rax
  __int64 v10; // rax
  __int64 v11; // rax
  __int64 *v12; // rax
  bool v13; // zf
  __int64 result; // rax
  __int64 *v16[4]; // [rsp+10h] [rbp-70h] BYREF
  _BYTE *v17; // [rsp+30h] [rbp-50h] BYREF
  __int16 v18; // [rsp+40h] [rbp-40h]

  v9 = sub_1643350(a1);
  v16[0] = (__int64 *)sub_159C470(v9, 0, 0);
  v10 = sub_1643350(a1);
  v16[1] = (__int64 *)sub_159C470(v10, 0, 0);
  v11 = sub_1643350(a1);
  v12 = (__int64 *)sub_159C470(v11, a5, 0);
  v13 = *a6 == 0;
  v16[2] = v12;
  v18 = 257;
  if ( !v13 )
  {
    v17 = a6;
    LOBYTE(v18) = 3;
  }
  result = sub_1BBF860(a2, a3, a4, v16, 3u, (__int64 *)&v17);
  if ( *(_BYTE *)(result + 16) != 56 )
    return 0;
  return result;
}
