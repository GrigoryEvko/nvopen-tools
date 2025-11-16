// Function: sub_39D0460
// Address: 0x39d0460
//
void __fastcall sub_39D0460(__int64 a1, _QWORD *a2)
{
  __int64 v2; // rax
  __int64 v3; // rax
  const char *v4; // rax
  __int64 v5; // rdx
  void (__fastcall *v6)(__int64, const char ***); // rax
  __int64 v7; // rax
  __int128 v8; // [rsp+0h] [rbp-80h] BYREF
  const char *v9; // [rsp+10h] [rbp-70h] BYREF
  __int64 v10; // [rsp+18h] [rbp-68h]
  _BYTE v11[16]; // [rsp+20h] [rbp-60h] BYREF
  const char **v12; // [rsp+30h] [rbp-50h] BYREF
  __int64 v13; // [rsp+38h] [rbp-48h]
  __int64 v14; // [rsp+40h] [rbp-40h]
  __int64 v15; // [rsp+48h] [rbp-38h]
  int v16; // [rsp+50h] [rbp-30h]
  __int128 *v17; // [rsp+58h] [rbp-28h]

  if ( (*(unsigned __int8 (__fastcall **)(__int64))(*(_QWORD *)a1 + 16LL))(a1) )
  {
    v11[0] = 0;
    v9 = v11;
    v10 = 0;
    v16 = 1;
    v12 = (const char **)&unk_49EFBE0;
    v15 = 0;
    v14 = 0;
    v13 = 0;
    v17 = (__int128 *)&v9;
    v7 = sub_16E4080(a1);
    sub_16E5B20(a2, v7, (__int64)&v12);
    if ( v15 != v13 )
      sub_16E7BA0((__int64 *)&v12);
    v8 = *v17;
    (*(void (__fastcall **)(__int64, __int128 *, _QWORD))(*(_QWORD *)a1 + 216LL))(a1, &v8, 0);
    sub_16E7BC0((__int64 *)&v12);
    if ( v9 != v11 )
      j_j___libc_free_0((unsigned __int64)v9);
  }
  else
  {
    v2 = *(_QWORD *)a1;
    v8 = 0u;
    (*(void (__fastcall **)(__int64, __int128 *, _QWORD))(v2 + 216))(a1, &v8, 0);
    v3 = sub_16E4080(a1);
    v4 = sub_16E5B30(v8, v3, a2);
    v10 = v5;
    v9 = v4;
    if ( v5 )
    {
      v6 = *(void (__fastcall **)(__int64, const char ***))(*(_QWORD *)a1 + 232LL);
      LOWORD(v14) = 261;
      v12 = &v9;
      v6(a1, &v12);
    }
  }
}
