// Function: sub_692440
// Address: 0x692440
//
__int64 __fastcall sub_692440(__int64 a1, __int64 a2)
{
  __int64 v2; // rax
  char i; // dl
  __int64 v4; // rax
  __int64 v5; // r12
  __int64 v7; // [rsp+8h] [rbp-4E8h] BYREF
  _BYTE v8[17]; // [rsp+10h] [rbp-4E0h] BYREF
  char v9; // [rsp+21h] [rbp-4CFh]
  __int64 v10[44]; // [rsp+B0h] [rbp-440h] BYREF
  __int64 v11[44]; // [rsp+210h] [rbp-2E0h] BYREF
  _QWORD v12[2]; // [rsp+370h] [rbp-180h] BYREF
  char v13; // [rsp+380h] [rbp-170h]

  sub_6E1DD0(&v7);
  sub_6E1E00(5, v8, 0, 0);
  v9 |= 3u;
  sub_6E7150(a1, v10);
  sub_6E7150(a2, v11);
  sub_6907F0(v10, v11, 0x2Fu, dword_4F07508, dword_4F06650[0], (__int64)v12);
  if ( v13 )
  {
    v2 = v12[0];
    for ( i = *(_BYTE *)(v12[0] + 140LL); i == 12; i = *(_BYTE *)(v2 + 140) )
      v2 = *(_QWORD *)(v2 + 160);
    if ( i )
      sub_688FA0(v12);
  }
  v4 = sub_6F6F40(v12, 0);
  v5 = sub_6E2700(v4);
  sub_6E2B30();
  sub_6E1DF0(v7);
  return v5;
}
