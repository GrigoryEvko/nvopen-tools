// Function: sub_215BA10
// Address: 0x215ba10
//
void *__fastcall sub_215BA10(__int64 a1, unsigned int a2, __int64 a3)
{
  __int64 v4; // r8
  __int64 (__fastcall *v5)(); // rax
  void *result; // rax
  _QWORD v7[2]; // [rsp+0h] [rbp-90h] BYREF
  _QWORD v8[2]; // [rsp+10h] [rbp-80h] BYREF
  char *v9[2]; // [rsp+20h] [rbp-70h] BYREF
  _QWORD v10[2]; // [rsp+30h] [rbp-60h] BYREF
  void *v11; // [rsp+40h] [rbp-50h] BYREF
  __int64 v12; // [rsp+48h] [rbp-48h]
  __int64 v13; // [rsp+50h] [rbp-40h]
  __int64 v14; // [rsp+58h] [rbp-38h]
  int v15; // [rsp+60h] [rbp-30h]
  _QWORD *v16; // [rsp+68h] [rbp-28h]

  v4 = *(_QWORD *)(a1 + 8);
  LOBYTE(v8[0]) = 0;
  v7[0] = v8;
  v15 = 1;
  v7[1] = 0;
  v14 = 0;
  v13 = 0;
  v12 = 0;
  v11 = &unk_49EFBE0;
  v16 = v7;
  v5 = *(__int64 (__fastcall **)())(*(_QWORD *)v4 + 392LL);
  if ( v5 == sub_215BB50 )
  {
    v9[0] = (char *)v10;
    sub_215B830((__int64 *)v9, "%ERROR", (__int64)"");
  }
  else
  {
    ((void (__fastcall *)(char **, __int64, _QWORD))v5)(v9, v4, a2);
  }
  sub_16E7EE0((__int64)&v11, v9[0], (size_t)v9[1]);
  if ( (_QWORD *)v9[0] != v10 )
    j_j___libc_free_0(v9[0], v10[0] + 1LL);
  if ( v14 != v12 )
  {
    sub_16E7BA0((__int64 *)&v11);
    if ( v14 != v12 )
      sub_16E7BA0((__int64 *)&v11);
  }
  sub_216EF30(*v16, a3);
  result = sub_16E7BC0((__int64 *)&v11);
  if ( (_QWORD *)v7[0] != v8 )
    return (void *)j_j___libc_free_0(v7[0], v8[0] + 1LL);
  return result;
}
