// Function: sub_17E5460
// Address: 0x17e5460
//
__int64 *__fastcall sub_17E5460(__int64 *a1, _QWORD *a2, __int64 *a3)
{
  __int64 v4; // r14
  __int64 v5; // r15
  __int64 *v6; // rax
  __int64 v7; // rax
  __int64 v8; // rax
  _QWORD *v10; // [rsp+0h] [rbp-90h] BYREF
  __int16 v11; // [rsp+10h] [rbp-80h]
  _QWORD v12[2]; // [rsp+20h] [rbp-70h] BYREF
  __int64 v13; // [rsp+30h] [rbp-60h] BYREF
  _QWORD v14[10]; // [rsp+40h] [rbp-50h] BYREF

  if ( (*(unsigned __int8 (__fastcall **)(_QWORD, void *))(*(_QWORD *)*a2 + 48LL))(*a2, &unk_4FA032B) )
  {
    v4 = *a2;
    *a2 = 0;
    v5 = *a3;
    (*(void (__fastcall **)(_QWORD *, __int64))(*(_QWORD *)v4 + 24LL))(v12, v4);
    v11 = 260;
    v6 = (__int64 *)a3[1];
    v10 = v12;
    v7 = *v6;
    v14[1] = 18;
    v14[2] = v7;
    v14[3] = &v10;
    v14[0] = &unk_49ECF40;
    sub_16027F0(v5, (__int64)v14);
    if ( (__int64 *)v12[0] != &v13 )
      j_j___libc_free_0(v12[0], v13 + 1);
    *a1 = 1;
    (*(void (__fastcall **)(__int64))(*(_QWORD *)v4 + 8LL))(v4);
  }
  else
  {
    v8 = *a2;
    *a2 = 0;
    *a1 = v8 | 1;
  }
  return a1;
}
