// Function: sub_2BE6250
// Address: 0x2be6250
//
__int64 __fastcall sub_2BE6250(__int64 a1, _QWORD *a2, const void *a3, __int64 a4)
{
  __int64 v6; // rax
  signed __int64 v7; // r12
  __int64 v8; // r8
  char *v9; // r13
  char *v10; // r9
  __int64 v11; // r14
  __int64 v13; // [rsp+0h] [rbp-60h]
  __int64 v14; // [rsp+8h] [rbp-58h]
  __int64 v15[2]; // [rsp+10h] [rbp-50h] BYREF
  _QWORD v16[8]; // [rsp+20h] [rbp-40h] BYREF

  v6 = sub_222F790(a2, (__int64)a2);
  v7 = a4 - (_QWORD)a3;
  if ( v7 < 0 )
    sub_4262D8((__int64)"cannot create std::vector larger than max_size()");
  v8 = v6;
  if ( v7 )
  {
    v13 = v6;
    v9 = (char *)sub_22077B0(v7);
    memcpy(v9, a3, v7);
    v10 = &v9[v7];
    v8 = v13;
  }
  else
  {
    v9 = 0;
    v10 = 0;
  }
  v14 = (__int64)v10;
  (*(void (__fastcall **)(__int64, char *, char *))(*(_QWORD *)v8 + 40LL))(v8, v9, v10);
  v11 = sub_221F880(a2, (__int64)v9);
  v15[0] = (__int64)v16;
  sub_2BDC2F0(v15, v9, v14);
  (*(void (__fastcall **)(__int64, __int64, __int64, __int64))(*(_QWORD *)v11 + 24LL))(a1, v11, v15[0], v15[0] + v15[1]);
  if ( (_QWORD *)v15[0] != v16 )
    j_j___libc_free_0(v15[0]);
  if ( v9 )
    j_j___libc_free_0((unsigned __int64)v9);
  return a1;
}
