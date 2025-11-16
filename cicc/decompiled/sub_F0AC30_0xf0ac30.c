// Function: sub_F0AC30
// Address: 0xf0ac30
//
_QWORD *__fastcall sub_F0AC30(_QWORD *a1, __int64 a2, __int64 a3)
{
  __int64 *v5; // r15
  const char *v6; // rax
  __int64 v7; // rdi
  int *v8; // rax
  __int64 v9; // rdx
  unsigned __int8 *v10; // r14
  __int64 v11; // r15
  __int64 *v12; // rax
  __int64 v13; // rax
  __int64 v14; // r13
  _QWORD *v15; // r12
  __int64 v17; // rdx
  int v18; // r13d
  __int64 v19; // rbx
  __int64 v20; // r13
  __int64 v21; // rdx
  unsigned int v22; // esi
  int v23; // [rsp+4h] [rbp-ACh]
  unsigned __int8 *v24; // [rsp+18h] [rbp-98h] BYREF
  _QWORD v25[4]; // [rsp+20h] [rbp-90h] BYREF
  __int16 v26; // [rsp+40h] [rbp-70h]
  _QWORD v27[4]; // [rsp+50h] [rbp-60h] BYREF
  __int16 v28; // [rsp+70h] [rbp-40h]

  v5 = *(__int64 **)(*a1 + 32LL);
  v6 = sub_BD5D20(a1[2]);
  v26 = 261;
  v7 = v5[10];
  v25[0] = v6;
  v8 = (int *)a1[1];
  v25[1] = v9;
  v23 = *v8;
  v10 = (unsigned __int8 *)(*(__int64 (__fastcall **)(__int64, _QWORD, __int64, __int64))(*(_QWORD *)v7 + 16LL))(
                             v7,
                             (unsigned int)*v8,
                             a2,
                             a3);
  if ( !v10 )
  {
    v28 = 257;
    v10 = (unsigned __int8 *)sub_B504D0(v23, a2, a3, (__int64)v27, 0, 0);
    if ( (unsigned __int8)sub_920620((__int64)v10) )
    {
      v17 = v5[12];
      v18 = *((_DWORD *)v5 + 26);
      if ( v17 )
        sub_B99FD0((__int64)v10, 3u, v17);
      sub_B45150((__int64)v10, v18);
    }
    (*(void (__fastcall **)(__int64, unsigned __int8 *, _QWORD *, __int64, __int64))(*(_QWORD *)v5[11] + 16LL))(
      v5[11],
      v10,
      v25,
      v5[7],
      v5[8]);
    v19 = *v5;
    v20 = *v5 + 16LL * *((unsigned int *)v5 + 2);
    if ( *v5 != v20 )
    {
      do
      {
        v21 = *(_QWORD *)(v19 + 8);
        v22 = *(_DWORD *)v19;
        v19 += 16;
        sub_B99FD0((__int64)v10, v22, v21);
      }
      while ( v20 != v19 );
    }
  }
  v24 = v10;
  if ( (unsigned __int8)(*v10 - 42) <= 0x11u )
    sub_B45260(v10, a1[2], 1);
  v11 = 0;
  v12 = (__int64 *)sub_B43CA0(a1[2]);
  v27[0] = *((_QWORD *)v24 + 1);
  v13 = sub_B6E160(v12, 0x192u, (__int64)v27, 1);
  v28 = 257;
  v14 = v13;
  if ( v13 )
    v11 = *(_QWORD *)(v13 + 24);
  v15 = sub_BD2CC0(88, 2u);
  if ( v15 )
  {
    sub_B44260((__int64)v15, **(_QWORD **)(v11 + 16), 56, 2u, 0, 0);
    v15[9] = 0;
    sub_B4A290((__int64)v15, v11, v14, (__int64 *)&v24, 1, (__int64)v27, 0, 0);
  }
  return v15;
}
