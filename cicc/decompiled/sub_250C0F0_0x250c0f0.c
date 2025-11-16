// Function: sub_250C0F0
// Address: 0x250c0f0
//
__int64 __fastcall sub_250C0F0(_QWORD *a1)
{
  unsigned int v1; // r13d
  _BYTE *v2; // rsi
  __int64 v3; // rdx
  _QWORD *v5; // [rsp+0h] [rbp-60h] BYREF
  _QWORD v6[2]; // [rsp+10h] [rbp-50h] BYREF
  __int64 v7; // [rsp+20h] [rbp-40h]
  __int64 v8; // [rsp+28h] [rbp-38h]
  __int64 v9; // [rsp+30h] [rbp-30h]

  v2 = (_BYTE *)a1[29];
  v3 = a1[30];
  v5 = v6;
  sub_2506C40((__int64 *)&v5, v2, (__int64)&v2[v3]);
  v7 = a1[33];
  v8 = a1[34];
  LOBYTE(v1) = (((_DWORD)v7 - 26) & 0xFFFFFFEE) == 0;
  v9 = a1[35];
  if ( v5 != v6 )
    j_j___libc_free_0((unsigned __int64)v5);
  return v1;
}
