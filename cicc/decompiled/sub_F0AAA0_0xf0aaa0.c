// Function: sub_F0AAA0
// Address: 0xf0aaa0
//
_QWORD *__fastcall sub_F0AAA0(_QWORD *a1, __int64 a2, __int64 a3, void *a4, __int64 a5)
{
  __int64 *v6; // r15
  int *v7; // rax
  unsigned __int8 *v8; // r14
  _QWORD *v9; // rax
  __int64 v10; // r9
  _QWORD *v11; // r12
  __int64 v13; // rdx
  int v14; // r12d
  __int64 v15; // r13
  __int64 v16; // r12
  __int64 v17; // rdx
  unsigned int v18; // esi
  int v19; // [rsp+Ch] [rbp-A4h]
  char v22[32]; // [rsp+20h] [rbp-90h] BYREF
  __int16 v23; // [rsp+40h] [rbp-70h]
  _BYTE v24[32]; // [rsp+50h] [rbp-60h] BYREF
  __int16 v25; // [rsp+70h] [rbp-40h]

  v6 = *(__int64 **)(*a1 + 32LL);
  v7 = (int *)a1[1];
  v23 = 257;
  v19 = *v7;
  v8 = (unsigned __int8 *)(*(__int64 (__fastcall **)(__int64, _QWORD, __int64, __int64))(*(_QWORD *)v6[10] + 16LL))(
                            v6[10],
                            (unsigned int)*v7,
                            a2,
                            a3);
  if ( !v8 )
  {
    v25 = 257;
    v8 = (unsigned __int8 *)sub_B504D0(v19, a2, a3, (__int64)v24, 0, 0);
    if ( (unsigned __int8)sub_920620((__int64)v8) )
    {
      v13 = v6[12];
      v14 = *((_DWORD *)v6 + 26);
      if ( v13 )
        sub_B99FD0((__int64)v8, 3u, v13);
      sub_B45150((__int64)v8, v14);
    }
    (*(void (__fastcall **)(__int64, unsigned __int8 *, char *, __int64, __int64))(*(_QWORD *)v6[11] + 16LL))(
      v6[11],
      v8,
      v22,
      v6[7],
      v6[8]);
    v15 = *v6;
    v16 = *v6 + 16LL * *((unsigned int *)v6 + 2);
    if ( *v6 != v16 )
    {
      do
      {
        v17 = *(_QWORD *)(v15 + 8);
        v18 = *(_DWORD *)v15;
        v15 += 16;
        sub_B99FD0((__int64)v8, v18, v17);
      }
      while ( v16 != v15 );
    }
  }
  if ( (unsigned __int8)(*v8 - 42) <= 0x11u )
    sub_B45260(v8, a1[2], 1);
  v25 = 257;
  v9 = sub_BD2C40(112, unk_3F1FE60);
  v11 = v9;
  if ( v9 )
    sub_B4EB40((__int64)v9, (__int64)v8, a4, a5, (__int64)v24, v10, 0);
  return v11;
}
