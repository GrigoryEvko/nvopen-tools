// Function: sub_1949540
// Address: 0x1949540
//
char __fastcall sub_1949540(__int64 a1, __int64 a2, __int64 a3, char a4)
{
  unsigned int v6; // eax
  unsigned int v7; // r15d
  __int64 v8; // rbx
  unsigned int v9; // r15d
  char result; // al
  __int64 v11; // rax
  char v12; // [rsp+Fh] [rbp-41h]
  __int64 v13; // [rsp+10h] [rbp-40h] BYREF
  unsigned int v14; // [rsp+18h] [rbp-38h]

  v6 = *(_DWORD *)(sub_1456040(a1) + 8) >> 8;
  v14 = v6;
  if ( a4 )
  {
    v7 = v6 - 1;
    v8 = 1LL << ((unsigned __int8)v6 - 1);
    if ( v6 <= 0x40 )
    {
      v13 = 0;
LABEL_4:
      v13 |= v8;
      v9 = 38;
      goto LABEL_7;
    }
    sub_16A4EF0((__int64)&v13, 0, 0);
    if ( v14 <= 0x40 )
      goto LABEL_4;
    *(_QWORD *)(v13 + 8LL * (v7 >> 6)) |= v8;
    v9 = 38;
  }
  else if ( v6 <= 0x40 )
  {
    v13 = 0;
    v9 = 34;
  }
  else
  {
    v9 = 34;
    sub_16A4EF0((__int64)&v13, 0, 0);
  }
LABEL_7:
  result = sub_146D950(a3, a1, a2);
  if ( result )
  {
    v11 = sub_145CF40(a3, (__int64)&v13);
    result = sub_148B410(a3, a2, v9, a1, v11);
  }
  if ( v14 > 0x40 )
  {
    if ( v13 )
    {
      v12 = result;
      j_j___libc_free_0_0(v13);
      return v12;
    }
  }
  return result;
}
