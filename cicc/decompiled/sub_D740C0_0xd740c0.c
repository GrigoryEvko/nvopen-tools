// Function: sub_D740C0
// Address: 0xd740c0
//
unsigned __int64 __fastcall sub_D740C0(__int64 *a1, __int64 a2, __int64 a3)
{
  _QWORD *v4; // rax
  _QWORD *v5; // rbx
  unsigned __int64 v6; // rax
  unsigned __int64 v7; // rax
  unsigned __int64 result; // rax
  __int64 v9; // [rsp+0h] [rbp-70h] BYREF
  __int64 v10; // [rsp+8h] [rbp-68h] BYREF
  __int64 v11; // [rsp+10h] [rbp-60h]
  unsigned __int64 v12; // [rsp+18h] [rbp-58h]
  char v13[80]; // [rsp+20h] [rbp-50h] BYREF

  v4 = (_QWORD *)sub_D68C20(*a1, a2);
  if ( !v4 )
    return sub_D72D40(a1, a2, a3);
  v5 = v4;
  v6 = *v4 & 0xFFFFFFFFFFFFFFF8LL;
  if ( v6 )
  {
    v7 = v6 - 48;
    v9 = a2;
    v10 = 6;
    v11 = 0;
    v12 = v7;
    if ( v7 != -8192 && v7 != -4096 )
      sub_BD73F0((__int64)&v10);
  }
  else
  {
    v9 = a2;
    v10 = 6;
    v11 = 0;
    v12 = 0;
  }
  sub_D6C550((__int64)v13, a3, &v9, &v10);
  sub_D68D70(&v10);
  result = 0;
  if ( (*v5 & 0xFFFFFFFFFFFFFFF8LL) != 0 )
    return (*v5 & 0xFFFFFFFFFFFFFFF8LL) - 48;
  return result;
}
