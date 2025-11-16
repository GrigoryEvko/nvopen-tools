// Function: sub_39D0990
// Address: 0x39d0990
//
void __fastcall sub_39D0990(__int64 a1, __int64 a2)
{
  __int64 v2; // rax
  __int64 v3; // rax
  __int64 v4; // r15
  __int64 v5; // rbx
  __int64 v6; // r14
  __int64 v7; // rax
  __int64 v8; // rdx
  __int64 v9; // rax
  __int64 v10; // rdx
  void (__fastcall *v11)(__int64, const char ***); // rax
  __int64 v12; // rax
  const char *v13; // [rsp+0h] [rbp-90h] BYREF
  const char *v14; // [rsp+8h] [rbp-88h]
  const char *v15; // [rsp+10h] [rbp-80h] BYREF
  __int64 v16; // [rsp+18h] [rbp-78h]
  _BYTE v17[16]; // [rsp+20h] [rbp-70h] BYREF
  const char **v18; // [rsp+30h] [rbp-60h] BYREF
  __int64 v19; // [rsp+38h] [rbp-58h]
  __int64 v20; // [rsp+40h] [rbp-50h]
  __int64 v21; // [rsp+48h] [rbp-48h]
  int v22; // [rsp+50h] [rbp-40h]
  const char **v23; // [rsp+58h] [rbp-38h]

  if ( (*(unsigned __int8 (__fastcall **)(__int64))(*(_QWORD *)a1 + 16LL))(a1) )
  {
    v17[0] = 0;
    v15 = v17;
    v16 = 0;
    v22 = 1;
    v18 = (const char **)&unk_49EFBE0;
    v21 = 0;
    v20 = 0;
    v19 = 0;
    v23 = &v15;
    v12 = sub_16E4080(a1);
    sub_16E59D0((unsigned int *)a2, v12, (__int64)&v18);
    if ( v21 != v19 )
      sub_16E7BA0((__int64 *)&v18);
    v13 = *v23;
    v14 = v23[1];
    (*(void (__fastcall **)(__int64, const char **, _QWORD))(*(_QWORD *)a1 + 216LL))(a1, &v13, 0);
    sub_16E7BC0((__int64 *)&v18);
    if ( v15 != v17 )
      j_j___libc_free_0((unsigned __int64)v15);
  }
  else
  {
    v2 = *(_QWORD *)a1;
    v13 = 0;
    v14 = 0;
    (*(void (__fastcall **)(__int64, const char **, _QWORD))(v2 + 216))(a1, &v13, 0);
    v3 = sub_16E4080(a1);
    v4 = (__int64)v13;
    v5 = (__int64)v14;
    v6 = v3;
    v7 = sub_16E4250(v3);
    if ( v7 )
    {
      v8 = *(_QWORD *)(v7 + 16);
      v9 = *(_QWORD *)(v7 + 24);
      *(_QWORD *)(a2 + 8) = v8;
      *(_QWORD *)(a2 + 16) = v9;
    }
    v15 = sub_16E59E0(v4, v5, v6, (_DWORD *)a2);
    v16 = v10;
    if ( v10 )
    {
      v11 = *(void (__fastcall **)(__int64, const char ***))(*(_QWORD *)a1 + 232LL);
      LOWORD(v20) = 261;
      v18 = &v15;
      v11(a1, &v18);
    }
  }
}
