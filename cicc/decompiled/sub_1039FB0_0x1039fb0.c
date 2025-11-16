// Function: sub_1039FB0
// Address: 0x1039fb0
//
__int64 __fastcall sub_1039FB0(__int64 a1, __int64 a2)
{
  unsigned int v3; // eax
  unsigned int v4; // r13d
  __int64 *v5; // r15
  unsigned __int8 *v6; // rsi
  __int64 v8; // rax
  char v9; // [rsp+Fh] [rbp-61h] BYREF
  _QWORD v10[2]; // [rsp+10h] [rbp-60h] BYREF
  __int64 v11; // [rsp+20h] [rbp-50h]
  __int64 *v12; // [rsp+30h] [rbp-40h] BYREF
  __int64 v13; // [rsp+38h] [rbp-38h]
  __int64 v14; // [rsp+40h] [rbp-30h]

  LOBYTE(v3) = sub_10394B0(**(_BYTE **)a1);
  if ( (_BYTE)v3 )
  {
    v4 = 0;
    sub_10397E0((__int64 *)a1, a2, **(_BYTE **)a1, "single", 6u);
  }
  else if ( (**(_BYTE **)a1 & 4) != 0 && (v4 = v3, sub_1039530(a1, *(_QWORD *)a1), sub_10394B0(**(_BYTE **)a1)) )
  {
    sub_10397E0((__int64 *)a1, a2, **(_BYTE **)a1, "single", 6u);
  }
  else
  {
    v5 = (__int64 *)sub_BD5C60(a2);
    v10[0] = 0;
    v10[1] = 0;
    v11 = 0;
    sub_9CA200((__int64)v10, 0, (_QWORD *)(a1 + 8));
    v6 = *(unsigned __int8 **)a1;
    v12 = 0;
    v13 = 0;
    v14 = 0;
    v9 = 1;
    v4 = sub_1039CC0(a1, v6, v5, (__int64)v10, (__int64)&v12, 0, &v9);
    if ( (_BYTE)v4 )
    {
      v8 = sub_B9C770(v5, v12, (__int64 *)((v13 - (__int64)v12) >> 3), 0, 1);
      sub_B99FD0(a2, 0x22u, v8);
    }
    else
    {
      sub_10397E0((__int64 *)a1, a2, 1, "indistinguishable", 0x11u);
    }
    if ( v12 )
      j_j___libc_free_0(v12, v14 - (_QWORD)v12);
    if ( v10[0] )
      j_j___libc_free_0(v10[0], v11 - v10[0]);
  }
  return v4;
}
