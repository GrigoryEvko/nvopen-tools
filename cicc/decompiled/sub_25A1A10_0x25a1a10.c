// Function: sub_25A1A10
// Address: 0x25a1a10
//
void __fastcall sub_25A1A10(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5)
{
  __int64 *v5; // r12
  _QWORD *v6; // rdi
  __int64 v7; // rax
  __int64 (*v8)(void); // rdx
  __int64 (__fastcall *v9)(__int64); // rax
  __int64 v10; // rbx
  __int64 v11; // rdx
  __int64 v12; // rax
  unsigned __int64 v13; // rsi
  __int64 v14; // rax
  __int64 v15; // [rsp+8h] [rbp-A8h] BYREF
  _QWORD v16[2]; // [rsp+10h] [rbp-A0h] BYREF
  void *v17; // [rsp+20h] [rbp-90h] BYREF
  __int64 *v18; // [rsp+28h] [rbp-88h]
  __int16 v19; // [rsp+30h] [rbp-80h]
  __int64 *v20[2]; // [rsp+40h] [rbp-70h] BYREF
  __int64 v21; // [rsp+50h] [rbp-60h] BYREF
  void *v22; // [rsp+60h] [rbp-50h] BYREF
  __int16 v23; // [rsp+80h] [rbp-30h]

  v15 = a1;
  v23 = 257;
  v17 = sub_CB7210(a1, a2, a3, a4, a5);
  v18 = &v15;
  v19 = 0;
  sub_CA0F50((__int64 *)v20, &v22);
  sub_25612C0((__int64 *)&v17, v20);
  v5 = v18;
  v6 = (_QWORD *)*v18;
  v7 = *(_QWORD *)*v18;
  v8 = *(__int64 (**)(void))(v7 + 24);
  if ( (char *)v8 == (char *)sub_2505E00 )
  {
    v9 = *(__int64 (__fastcall **)(__int64))(v7 + 16);
    v10 = *(_QWORD *)(*(_QWORD *)(v6[1] + 200LL) + 32LL) + 8LL * *(unsigned int *)(*(_QWORD *)(v6[1] + 200LL) + 40LL);
    if ( v9 == sub_2505DF0 )
    {
LABEL_3:
      v11 = v6[1];
      v12 = *(_QWORD *)(*(_QWORD *)(v11 + 200) + 32LL);
      goto LABEL_4;
    }
  }
  else
  {
    v14 = v8();
    v6 = (_QWORD *)*v5;
    v10 = v14;
    v9 = *(__int64 (__fastcall **)(__int64))(*(_QWORD *)*v5 + 16LL);
    if ( v9 == sub_2505DF0 )
      goto LABEL_3;
  }
  v12 = ((__int64 (*)(void))v9)();
LABEL_4:
  v16[0] = v12;
  for ( v16[1] = v11; v16[0] != v10; v16[0] += 8LL )
  {
    v13 = sub_25A1010((__int64)v16);
    if ( v13 != *v18 )
      sub_25A1070((__int64)&v17, v13);
  }
  sub_904010((__int64)v17, "}\n");
  if ( v20[0] != &v21 )
    j_j___libc_free_0((unsigned __int64)v20[0]);
}
