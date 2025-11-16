// Function: sub_F70730
// Address: 0xf70730
//
char __fastcall sub_F70730(__int64 a1, __int64 a2, __int64 *a3, char a4)
{
  unsigned int v6; // eax
  unsigned int v7; // r15d
  __int64 v8; // rbx
  unsigned __int64 v9; // rdx
  unsigned int v10; // ebx
  char result; // al
  unsigned __int64 v12; // rdx
  _QWORD *v13; // rax
  char v14; // [rsp+Fh] [rbp-41h]
  unsigned __int64 v15; // [rsp+10h] [rbp-40h] BYREF
  unsigned int v16; // [rsp+18h] [rbp-38h]

  v6 = *(_DWORD *)(sub_D95540(a1) + 8) >> 8;
  v16 = v6;
  if ( a4 )
  {
    v7 = v6 - 1;
    v8 = ~(1LL << ((unsigned __int8)v6 - 1));
    if ( v6 <= 0x40 )
    {
      v9 = 0xFFFFFFFFFFFFFFFFLL >> -(char)v6;
      if ( !v6 )
        v9 = 0;
      v15 = v9;
      goto LABEL_6;
    }
    sub_C43690((__int64)&v15, -1, 1);
    if ( v16 <= 0x40 )
    {
LABEL_6:
      v15 &= v8;
      v10 = 40;
      goto LABEL_7;
    }
    *(_QWORD *)(v15 + 8LL * (v7 >> 6)) &= v8;
    v10 = 40;
  }
  else if ( v6 > 0x40 )
  {
    v10 = 36;
    sub_C43690((__int64)&v15, -1, 1);
  }
  else
  {
    v10 = 36;
    v12 = 0xFFFFFFFFFFFFFFFFLL >> -(char)v6;
    if ( !v6 )
      v12 = 0;
    v15 = v12;
  }
LABEL_7:
  result = sub_DAEB70((__int64)a3, a1, a2);
  if ( result )
  {
    v13 = sub_DA26C0(a3, (__int64)&v15);
    result = sub_DDD5B0(a3, a2, v10, a1, (__int64)v13);
  }
  if ( v16 > 0x40 )
  {
    if ( v15 )
    {
      v14 = result;
      j_j___libc_free_0_0(v15);
      return v14;
    }
  }
  return result;
}
