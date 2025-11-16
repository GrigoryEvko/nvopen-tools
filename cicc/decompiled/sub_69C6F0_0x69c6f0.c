// Function: sub_69C6F0
// Address: 0x69c6f0
//
__int64 __fastcall sub_69C6F0(unsigned __int16 a1, __int64 a2, __int64 a3)
{
  __int64 v4; // rdx
  __int64 v5; // rcx
  __int64 v6; // r8
  __int64 v7; // r9
  __int64 v8; // rax
  char i; // dl
  __int64 v10; // rdi
  __int64 v11; // r12
  __int64 v13; // [rsp+0h] [rbp-7D0h]
  __int64 v14; // [rsp+10h] [rbp-7C0h] BYREF
  __int64 v15; // [rsp+18h] [rbp-7B8h] BYREF
  _BYTE v16[17]; // [rsp+20h] [rbp-7B0h] BYREF
  char v17; // [rsp+31h] [rbp-79Fh]
  __int64 v18[44]; // [rsp+C0h] [rbp-710h] BYREF
  __int64 v19[44]; // [rsp+220h] [rbp-5B0h] BYREF
  __int64 v20[44]; // [rsp+380h] [rbp-450h] BYREF
  __int64 v21[44]; // [rsp+4E0h] [rbp-2F0h] BYREF
  _QWORD v22[2]; // [rsp+640h] [rbp-190h] BYREF
  char v23; // [rsp+650h] [rbp-180h]

  sub_6E1DD0(&v14);
  sub_6E1E00(5, v16, 0, 0);
  v17 |= 3u;
  sub_6E7150(a2, v18);
  sub_6E7150(a3, v19);
  sub_68FEF0(v18, v19, dword_4F07508, dword_4F06650[0], 0, (__int64)v20);
  sub_6907F0(v18, v19, 0x2Fu, dword_4F07508, dword_4F06650[0], (__int64)v22);
  v13 = sub_72BA30(5);
  v15 = sub_724DC0(5, v19, v4, v5, v6, v7);
  sub_72BB40(v13, v15);
  sub_6E6A50(v15, v21);
  sub_724E30(&v15);
  sub_69B310(v20, v21, a1, dword_4F07508, dword_4F06650[0], (__int64)v22);
  if ( v23 )
  {
    v8 = v22[0];
    for ( i = *(_BYTE *)(v22[0] + 140LL); i == 12; i = *(_BYTE *)(v8 + 140) )
      v8 = *(_QWORD *)(v8 + 160);
    if ( i )
      sub_688FA0(v22);
  }
  v10 = sub_6F6F40(v22, 0);
  v11 = sub_6E2700(v10);
  sub_6E2B30(v10, 0);
  sub_6E1DF0(v14);
  return v11;
}
