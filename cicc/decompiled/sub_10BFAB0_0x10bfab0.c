// Function: sub_10BFAB0
// Address: 0x10bfab0
//
__int64 *__fastcall sub_10BFAB0(__int64 a1, __int64 a2, unsigned __int8 *a3)
{
  __int64 v5; // r14
  __int64 *v6; // rbx
  const char *v7; // rax
  __int64 v8; // rdi
  __int64 v9; // rdx
  __int64 *v10; // r14
  __int64 v12; // rax
  __int64 v13; // rbx
  __int64 v14; // rbx
  __int64 v15; // rdx
  unsigned int v16; // esi
  __int64 v18; // [rsp+8h] [rbp-98h]
  __int64 v19; // [rsp+8h] [rbp-98h]
  _QWORD v20[4]; // [rsp+10h] [rbp-90h] BYREF
  __int16 v21; // [rsp+30h] [rbp-70h]
  __int64 *v22; // [rsp+40h] [rbp-60h] BYREF
  __int64 v23; // [rsp+48h] [rbp-58h]
  __int16 v24; // [rsp+60h] [rbp-40h]

  v5 = *(_QWORD *)(a1 + 32);
  sub_B445D0((__int64)&v22, (char *)a2);
  sub_10BF960(v5, (__int64)v22, v23);
  v6 = *(__int64 **)(a1 + 32);
  v7 = sub_BD5D20(a2);
  v8 = *(_QWORD *)(a2 + 8);
  v20[0] = v7;
  v21 = 773;
  v20[1] = v9;
  v20[2] = ".not";
  v18 = sub_AD62B0(v8);
  v10 = (__int64 *)(*(__int64 (__fastcall **)(__int64, __int64, __int64, __int64))(*(_QWORD *)v6[10] + 16LL))(
                     v6[10],
                     30,
                     a2,
                     v18);
  if ( !v10 )
  {
    v24 = 257;
    v10 = (__int64 *)sub_B504D0(30, a2, v18, (__int64)&v22, 0, 0);
    (*(void (__fastcall **)(__int64, __int64 *, _QWORD *, __int64, __int64))(*(_QWORD *)v6[11] + 16LL))(
      v6[11],
      v10,
      v20,
      v6[7],
      v6[8]);
    v12 = *v6;
    v13 = 16LL * *((unsigned int *)v6 + 2);
    v19 = v12 + v13;
    if ( v12 != v12 + v13 )
    {
      v14 = v12;
      do
      {
        v15 = *(_QWORD *)(v14 + 8);
        v16 = *(_DWORD *)v14;
        v14 += 16;
        sub_B99FD0((__int64)v10, v16, v15);
      }
      while ( v19 != v14 );
    }
  }
  v22 = v10;
  sub_BD79D0((__int64 *)a2, v10, (unsigned __int8 (__fastcall *)(__int64, __int64 *))sub_10B8250, (__int64)&v22);
  sub_F16650(a1, (__int64)v10, a3);
  return v10;
}
