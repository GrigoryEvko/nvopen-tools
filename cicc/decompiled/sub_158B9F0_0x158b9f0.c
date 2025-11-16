// Function: sub_158B9F0
// Address: 0x158b9f0
//
char __fastcall sub_158B9F0(__int64 a1)
{
  unsigned int v1; // eax
  unsigned int v2; // ebx
  __int64 v3; // r12
  char result; // al
  unsigned int v5; // eax
  unsigned int v6; // ebx
  __int64 v7; // r12
  char v8; // [rsp+Fh] [rbp-41h]
  char v9; // [rsp+Fh] [rbp-41h]
  unsigned __int64 v10; // [rsp+10h] [rbp-40h] BYREF
  unsigned int v11; // [rsp+18h] [rbp-38h]
  __int64 v12; // [rsp+20h] [rbp-30h] BYREF
  unsigned int v13; // [rsp+28h] [rbp-28h]

  v1 = *(_DWORD *)(a1 + 8);
  v2 = v1 - 1;
  v11 = v1;
  v3 = ~(1LL << ((unsigned __int8)v1 - 1));
  if ( v1 <= 0x40 )
  {
    v10 = 0xFFFFFFFFFFFFFFFFLL >> -(char)v1;
    goto LABEL_3;
  }
  sub_16A4EF0(&v10, -1, 1);
  if ( v11 <= 0x40 )
  {
LABEL_3:
    v10 &= v3;
    result = sub_158B950(a1, (__int64)&v10);
    if ( !result )
      goto LABEL_4;
    goto LABEL_10;
  }
  *(_QWORD *)(v10 + 8LL * (v2 >> 6)) &= v3;
  result = sub_158B950(a1, (__int64)&v10);
  if ( !result )
    goto LABEL_4;
LABEL_10:
  v5 = *(_DWORD *)(a1 + 8);
  v6 = v5 - 1;
  v13 = v5;
  v7 = 1LL << ((unsigned __int8)v5 - 1);
  if ( v5 <= 0x40 )
  {
    v12 = 0;
    goto LABEL_17;
  }
  sub_16A4EF0(&v12, 0, 0);
  if ( v13 <= 0x40 )
  {
LABEL_17:
    v12 |= v7;
    goto LABEL_13;
  }
  *(_QWORD *)(v12 + 8LL * (v6 >> 6)) |= v7;
LABEL_13:
  result = sub_158B950(a1, (__int64)&v12);
  if ( v13 > 0x40 && v12 )
  {
    v9 = result;
    j_j___libc_free_0_0(v12);
    result = v9;
  }
LABEL_4:
  if ( v11 > 0x40 )
  {
    if ( v10 )
    {
      v8 = result;
      j_j___libc_free_0_0(v10);
      return v8;
    }
  }
  return result;
}
