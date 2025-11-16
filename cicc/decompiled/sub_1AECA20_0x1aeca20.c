// Function: sub_1AECA20
// Address: 0x1aeca20
//
void __fastcall sub_1AECA20(__int64 a1, __int64 a2, __int64 a3, __int64 *a4)
{
  unsigned int v6; // ebx
  char v7; // bl
  __int64 *v8; // rax
  __int64 v9; // rax
  __int64 v10; // [rsp+0h] [rbp-60h] BYREF
  unsigned int v11; // [rsp+8h] [rbp-58h]
  __int64 v12; // [rsp+10h] [rbp-50h] BYREF
  unsigned int v13; // [rsp+18h] [rbp-48h]
  __int64 v14; // [rsp+20h] [rbp-40h]
  unsigned int v15; // [rsp+28h] [rbp-38h]

  if ( *(_BYTE *)(*a4 + 8) == 15 )
  {
    v6 = sub_15A95F0(a1, *a4);
    sub_1593050((__int64)&v12, a3);
    v11 = v6;
    if ( v6 <= 0x40 )
      v10 = 0;
    else
      sub_16A4EF0((__int64)&v10, 0, 0);
    v7 = sub_158B950((__int64)&v12, (__int64)&v10);
    if ( v11 > 0x40 && v10 )
      j_j___libc_free_0_0(v10);
    if ( v15 > 0x40 && v14 )
      j_j___libc_free_0_0(v14);
    if ( v13 > 0x40 && v12 )
      j_j___libc_free_0_0(v12);
    if ( !v7 )
    {
      v8 = (__int64 *)sub_16498A0(a2);
      v9 = sub_1627350(v8, 0, 0, 0, 1);
      sub_1625C10((__int64)a4, 11, v9);
    }
  }
}
