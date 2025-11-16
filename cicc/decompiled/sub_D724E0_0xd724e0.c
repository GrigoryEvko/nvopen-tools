// Function: sub_D724E0
// Address: 0xd724e0
//
__int64 __fastcall sub_D724E0(__int64 *a1, __int64 a2, __int64 a3, __int64 a4)
{
  char *v4; // rax
  char *v5; // rax
  _QWORD v7[2]; // [rsp+0h] [rbp-2D0h] BYREF
  char v8; // [rsp+10h] [rbp-2C0h] BYREF
  _QWORD v9[2]; // [rsp+130h] [rbp-1A0h] BYREF
  char v10; // [rsp+140h] [rbp-190h] BYREF
  char v11; // [rsp+260h] [rbp-70h] BYREF
  char *v12; // [rsp+268h] [rbp-68h]
  __int64 v13; // [rsp+270h] [rbp-60h]
  char v14; // [rsp+278h] [rbp-58h] BYREF

  v4 = &v8;
  v7[0] = 0;
  v7[1] = 1;
  do
  {
    *(_QWORD *)v4 = -4096;
    v4 += 72;
  }
  while ( v4 != (char *)v9 );
  v9[0] = 0;
  v5 = &v10;
  v9[1] = 1;
  do
  {
    *(_QWORD *)v5 = -4096;
    v5 += 72;
  }
  while ( v5 != &v11 );
  v11 = 0;
  v12 = &v14;
  v13 = 0x400000000LL;
  sub_D6FF50(a1, a2, a3, a4, (__int64)v7, a3);
  return sub_B1A8B0((__int64)v7, a2);
}
