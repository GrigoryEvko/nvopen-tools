// Function: sub_24A72E0
// Address: 0x24a72e0
//
__int64 *__fastcall sub_24A72E0(__int64 *a1, _QWORD *a2, __int64 a3)
{
  __int64 v4; // r14
  __int64 v5; // r15
  __int64 v6; // rax
  __int64 v7; // rax
  unsigned __int64 v9[2]; // [rsp+0h] [rbp-A0h] BYREF
  __int64 v10; // [rsp+10h] [rbp-90h] BYREF
  _QWORD v11[4]; // [rsp+20h] [rbp-80h] BYREF
  unsigned __int64 *v12; // [rsp+40h] [rbp-60h] BYREF
  __int16 v13; // [rsp+60h] [rbp-40h]

  if ( (*(unsigned __int8 (__fastcall **)(_QWORD, void *))(*(_QWORD *)*a2 + 48LL))(*a2, &unk_4F84053) )
  {
    v4 = *a2;
    *a2 = 0;
    v5 = *(_QWORD *)a3;
    (*(void (__fastcall **)(unsigned __int64 *, __int64))(*(_QWORD *)v4 + 24LL))(v9, v4);
    v12 = v9;
    v13 = 260;
    v6 = **(_QWORD **)(a3 + 8);
    v11[1] = 23;
    v11[2] = v6;
    v11[3] = &v12;
    v11[0] = &unk_49D9CA8;
    sub_B6EB20(v5, (__int64)v11);
    if ( (__int64 *)v9[0] != &v10 )
      j_j___libc_free_0(v9[0]);
    *a1 = 1;
    (*(void (__fastcall **)(__int64))(*(_QWORD *)v4 + 8LL))(v4);
  }
  else
  {
    v7 = *a2;
    *a2 = 0;
    *a1 = v7 | 1;
  }
  return a1;
}
