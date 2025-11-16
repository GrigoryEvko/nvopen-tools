// Function: sub_E47F60
// Address: 0xe47f60
//
__int64 *__fastcall sub_E47F60(__int64 *a1, __int64 *a2, __int64 a3)
{
  __int64 v4; // r14
  __int64 (__fastcall *v5)(__int64, __int64); // rdx
  __int64 v6; // rax
  __int64 v8; // [rsp+18h] [rbp-B8h]
  _BYTE v9[32]; // [rsp+20h] [rbp-B0h] BYREF
  _QWORD v10[2]; // [rsp+40h] [rbp-90h] BYREF
  _QWORD v11[2]; // [rsp+50h] [rbp-80h] BYREF
  _QWORD v12[4]; // [rsp+60h] [rbp-70h] BYREF
  __int64 v13; // [rsp+80h] [rbp-50h]
  __int64 v14; // [rsp+88h] [rbp-48h]
  _QWORD *v15; // [rsp+90h] [rbp-40h]

  if ( (*(unsigned __int8 (__fastcall **)(__int64, void *))(*(_QWORD *)*a2 + 48LL))(*a2, &unk_4F84053) )
  {
    v4 = *a2;
    *a2 = 0;
    v8 = **(_QWORD **)a3;
    v5 = *(__int64 (__fastcall **)(__int64, __int64))(*(_QWORD *)v4 + 24LL);
    if ( v5 == sub_9C3610 )
    {
      v15 = v10;
      LOBYTE(v11[0]) = 0;
      v14 = 0x100000000LL;
      v12[0] = &unk_49DD210;
      v10[0] = v11;
      v10[1] = 0;
      memset(&v12[1], 0, 24);
      v13 = 0;
      sub_CB5980((__int64)v12, 0, 0, 0);
      (*(void (__fastcall **)(__int64, _QWORD *))(*(_QWORD *)v4 + 16LL))(v4, v12);
      v12[0] = &unk_49DD210;
      sub_CB5840((__int64)v12);
    }
    else
    {
      v5((__int64)v10, v4);
    }
    v12[0] = v10;
    LOWORD(v13) = 260;
    sub_1061A30(v9, 0, v12);
    sub_B6EB20(v8, (__int64)v9);
    if ( (_QWORD *)v10[0] != v11 )
      j_j___libc_free_0(v10[0], v11[0] + 1LL);
    **(_BYTE **)(a3 + 8) = 1;
    *a1 = 1;
    (*(void (__fastcall **)(__int64))(*(_QWORD *)v4 + 8LL))(v4);
  }
  else
  {
    v6 = *a2;
    *a2 = 0;
    *a1 = v6 | 1;
  }
  return a1;
}
