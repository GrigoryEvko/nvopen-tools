// Function: sub_1C14360
// Address: 0x1c14360
//
_QWORD *__fastcall sub_1C14360(__int64 a1, _BYTE *a2)
{
  __int64 v2; // rax
  __int64 v3; // rax
  _QWORD *result; // rax
  __int64 v5; // rdx
  __int64 (__fastcall *v6)(__int64, _QWORD **); // rax
  __int64 v7; // rax
  __int64 v8; // [rsp+0h] [rbp-80h] BYREF
  __int64 v9; // [rsp+8h] [rbp-78h]
  _QWORD *v10; // [rsp+10h] [rbp-70h] BYREF
  __int64 v11; // [rsp+18h] [rbp-68h]
  _QWORD v12[2]; // [rsp+20h] [rbp-60h] BYREF
  _QWORD *v13; // [rsp+30h] [rbp-50h] BYREF
  __int64 v14; // [rsp+38h] [rbp-48h]
  __int64 v15; // [rsp+40h] [rbp-40h]
  __int64 v16; // [rsp+48h] [rbp-38h]
  int v17; // [rsp+50h] [rbp-30h]
  __int64 *v18; // [rsp+58h] [rbp-28h]

  if ( (*(unsigned __int8 (__fastcall **)(__int64))(*(_QWORD *)a1 + 16LL))(a1) )
  {
    LOBYTE(v12[0]) = 0;
    v10 = v12;
    v11 = 0;
    v17 = 1;
    v13 = &unk_49EFBE0;
    v16 = 0;
    v15 = 0;
    v14 = 0;
    v18 = (__int64 *)&v10;
    v7 = sub_16E4080(a1);
    sub_16E5680(a2, v7, (__int64)&v13);
    if ( v16 != v14 )
      sub_16E7BA0((__int64 *)&v13);
    v8 = *v18;
    v9 = v18[1];
    (*(void (__fastcall **)(__int64, __int64 *, _QWORD))(*(_QWORD *)a1 + 216LL))(a1, &v8, 0);
    result = sub_16E7BC0((__int64 *)&v13);
    if ( v10 != v12 )
      return (_QWORD *)j_j___libc_free_0(v10, v12[0] + 1LL);
  }
  else
  {
    v2 = *(_QWORD *)a1;
    v8 = 0;
    v9 = 0;
    (*(void (__fastcall **)(__int64, __int64 *, _QWORD))(v2 + 216))(a1, &v8, 0);
    v3 = sub_16E4080(a1);
    result = sub_16E5770(v8, v9, v3, a2);
    v11 = v5;
    v10 = result;
    if ( v5 )
    {
      v6 = *(__int64 (__fastcall **)(__int64, _QWORD **))(*(_QWORD *)a1 + 232LL);
      LOWORD(v15) = 261;
      v13 = &v10;
      return (_QWORD *)v6(a1, &v13);
    }
  }
  return result;
}
