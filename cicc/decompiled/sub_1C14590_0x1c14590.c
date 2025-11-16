// Function: sub_1C14590
// Address: 0x1c14590
//
_QWORD *__fastcall sub_1C14590(__int64 a1, int *a2)
{
  __int64 v2; // rax
  __int64 v3; // rax
  _QWORD *result; // rax
  __int64 v5; // rdx
  __int64 (__fastcall *v6)(__int64, _QWORD **); // rax
  __int64 v7; // rax
  __int128 v8; // [rsp+0h] [rbp-80h] BYREF
  _QWORD *v9; // [rsp+10h] [rbp-70h] BYREF
  __int64 v10; // [rsp+18h] [rbp-68h]
  _QWORD v11[2]; // [rsp+20h] [rbp-60h] BYREF
  _QWORD *v12; // [rsp+30h] [rbp-50h] BYREF
  __int64 v13; // [rsp+38h] [rbp-48h]
  __int64 v14; // [rsp+40h] [rbp-40h]
  __int64 v15; // [rsp+48h] [rbp-38h]
  int v16; // [rsp+50h] [rbp-30h]
  __int128 *v17; // [rsp+58h] [rbp-28h]

  if ( (*(unsigned __int8 (__fastcall **)(__int64))(*(_QWORD *)a1 + 16LL))(a1) )
  {
    LOBYTE(v11[0]) = 0;
    v9 = v11;
    v10 = 0;
    v16 = 1;
    v12 = &unk_49EFBE0;
    v15 = 0;
    v14 = 0;
    v13 = 0;
    v17 = (__int128 *)&v9;
    v7 = sub_16E4080(a1);
    sub_16E5AA0(a2, v7, (__int64)&v12);
    if ( v15 != v13 )
      sub_16E7BA0((__int64 *)&v12);
    v8 = *v17;
    (*(void (__fastcall **)(__int64, __int128 *, _QWORD))(*(_QWORD *)a1 + 216LL))(a1, &v8, 0);
    result = sub_16E7BC0((__int64 *)&v12);
    if ( v9 != v11 )
      return (_QWORD *)j_j___libc_free_0(v9, v11[0] + 1LL);
  }
  else
  {
    v2 = *(_QWORD *)a1;
    v8 = 0u;
    (*(void (__fastcall **)(__int64, __int128 *, _QWORD))(v2 + 216))(a1, &v8, 0);
    v3 = sub_16E4080(a1);
    result = sub_16E5AB0(v8, v3, a2);
    v10 = v5;
    v9 = result;
    if ( v5 )
    {
      v6 = *(__int64 (__fastcall **)(__int64, _QWORD **))(*(_QWORD *)a1 + 232LL);
      LOWORD(v14) = 261;
      v12 = &v9;
      return (_QWORD *)v6(a1, &v12);
    }
  }
  return result;
}
