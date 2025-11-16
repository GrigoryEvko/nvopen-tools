// Function: sub_6933C0
// Address: 0x6933c0
//
__int64 __fastcall sub_6933C0(__int64 a1, __int64 *a2, __int64 *a3)
{
  __int64 v3; // r12
  __int64 v4; // rdi
  __int64 v6; // rax
  __int64 v7; // rdi
  unsigned int v8; // [rsp+1Ch] [rbp-3D4h] BYREF
  _BYTE v9[64]; // [rsp+20h] [rbp-3D0h] BYREF
  _BYTE v10[160]; // [rsp+60h] [rbp-390h] BYREF
  __int64 v11[44]; // [rsp+100h] [rbp-2F0h] BYREF
  __m128i v12[25]; // [rsp+260h] [rbp-190h] BYREF

  v3 = *a3;
  sub_6E1E00(4, v10, 0, 0);
  if ( (*(_BYTE *)(a1 - 8) & 1) != 0 )
  {
    sub_7296C0(&v8);
    sub_6F8E70(a1, &dword_4F063F8, &dword_4F063F8, v11, 0);
    sub_878710(v3, v9);
    sub_82FD20((unsigned int)v11, 0, v3, v3, 0, 1, (__int64)&dword_4F063F8);
    sub_68D540(v11, v11[0], 0, 1u, (__int64)v9, (__int64)&dword_4F063F8, (__int64)&qword_4F063F0, 0, v12);
    *((_BYTE *)a2 + 177) = 5;
    v6 = sub_6F6F40(v12, 0);
    v7 = v8;
    a2[23] = v6;
    sub_729730(v7);
  }
  else
  {
    sub_6F8E70(a1, &dword_4F063F8, &dword_4F063F8, v11, 0);
    sub_878710(v3, v9);
    sub_82FD20((unsigned int)v11, 0, v3, v3, 0, 1, (__int64)&dword_4F063F8);
    sub_68D540(v11, v11[0], 0, 1u, (__int64)v9, (__int64)&dword_4F063F8, (__int64)&qword_4F063F0, 0, v12);
    *((_BYTE *)a2 + 177) = 5;
    a2[23] = sub_6F6F40(v12, 0);
  }
  v4 = *a2;
  sub_8756B0(*a2);
  return sub_6E2B30(v4, 0);
}
