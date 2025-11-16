// Function: sub_2114100
// Address: 0x2114100
//
__int64 __fastcall sub_2114100(_QWORD *a1, __int64 *a2, __int64 a3, _BYTE *a4, int a5, _BYTE *a6)
{
  __int64 v9; // rax
  __int64 v10; // rax
  __int64 *v11; // rax
  bool v12; // zf
  __int64 result; // rax
  __int64 *v15[2]; // [rsp+10h] [rbp-60h] BYREF
  _BYTE *v16; // [rsp+20h] [rbp-50h] BYREF
  __int16 v17; // [rsp+30h] [rbp-40h]

  v9 = sub_1643350(a1);
  v15[0] = (__int64 *)sub_159C470(v9, 0, 0);
  v10 = sub_1643350(a1);
  v11 = (__int64 *)sub_159C470(v10, a5, 0);
  v12 = *a6 == 0;
  v15[1] = v11;
  v17 = 257;
  if ( !v12 )
  {
    v16 = a6;
    LOBYTE(v17) = 3;
  }
  result = sub_1BBF860(a2, a3, a4, v15, 2u, (__int64 *)&v16);
  if ( *(_BYTE *)(result + 16) != 56 )
    return 0;
  return result;
}
