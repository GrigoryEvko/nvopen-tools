// Function: sub_D902F0
// Address: 0xd902f0
//
__int64 *__fastcall sub_D902F0(__int64 *a1, __int64 a2, __int64 a3, __int64 a4)
{
  _QWORD v6[2]; // [rsp+0h] [rbp-40h] BYREF
  __int64 (__fastcall *v7)(_QWORD *, _QWORD *, int); // [rsp+10h] [rbp-30h]
  __int64 (__fastcall *v8)(__int64 *, __int64); // [rsp+18h] [rbp-28h]

  v6[0] = *(_QWORD *)(sub_BC0510(a4, &unk_4F82418, a3) + 8);
  v8 = sub_D85630;
  v7 = sub_D85820;
  sub_D90260(a1, a3, (__int64)v6, 0);
  if ( v7 )
    v7(v6, v6, 3);
  return a1;
}
