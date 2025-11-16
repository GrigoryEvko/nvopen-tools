// Function: sub_F70610
// Address: 0xf70610
//
char __fastcall sub_F70610(__int64 a1, __int64 a2, __int64 *a3, char a4)
{
  unsigned int v6; // eax
  unsigned int v7; // ebx
  __int64 v8; // r15
  unsigned int v9; // ebx
  char result; // al
  _QWORD *v11; // rax
  char v12; // [rsp+Fh] [rbp-41h]
  __int64 v13; // [rsp+10h] [rbp-40h] BYREF
  unsigned int v14; // [rsp+18h] [rbp-38h]

  v6 = *(_DWORD *)(sub_D95540(a1) + 8) >> 8;
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
      goto LABEL_5;
    }
    sub_C43690((__int64)&v13, 0, 0);
    if ( v14 <= 0x40 )
      goto LABEL_4;
    *(_QWORD *)(v13 + 8LL * (v7 >> 6)) |= v8;
    v9 = 38;
  }
  else if ( v6 > 0x40 )
  {
    v9 = 34;
    sub_C43690((__int64)&v13, 0, 0);
  }
  else
  {
    v13 = 0;
    v9 = 34;
  }
LABEL_5:
  result = sub_DAEB70((__int64)a3, a1, a2);
  if ( result )
  {
    v11 = sub_DA26C0(a3, (__int64)&v13);
    result = sub_DDD5B0(a3, a2, v9, a1, (__int64)v11);
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
