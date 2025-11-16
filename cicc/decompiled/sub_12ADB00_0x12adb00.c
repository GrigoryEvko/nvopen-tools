// Function: sub_12ADB00
// Address: 0x12adb00
//
__int64 __fastcall sub_12ADB00(__int64 a1, __int64 a2, __int64 a3, unsigned int a4)
{
  __int64 v6; // r13
  char *v7; // r9
  __int64 v8; // rax
  __int64 v9; // rax
  __int64 v10; // r9
  __int64 v11; // rax
  __int64 *v12; // rdi
  __int64 v13; // rsi
  __int64 v14; // rax
  __int64 v15; // rax
  _BYTE *v16; // rdi
  __int64 v18; // [rsp+8h] [rbp-108h]
  char *v19; // [rsp+8h] [rbp-108h]
  _BYTE v21[16]; // [rsp+20h] [rbp-F0h] BYREF
  __int16 v22; // [rsp+30h] [rbp-E0h]
  _BYTE *v23; // [rsp+40h] [rbp-D0h] BYREF
  __int64 v24; // [rsp+48h] [rbp-C8h]
  _BYTE v25[64]; // [rsp+50h] [rbp-C0h] BYREF
  _BYTE *i; // [rsp+90h] [rbp-80h] BYREF
  __int64 v27; // [rsp+98h] [rbp-78h]
  _BYTE v28[112]; // [rsp+A0h] [rbp-70h] BYREF

  v6 = *(_QWORD *)(a3 + 16);
  v24 = 0x800000000LL;
  v27 = 0x800000000LL;
  v23 = v25;
  for ( i = v28; v6; LODWORD(v27) = v27 + 1 )
  {
    v7 = sub_128F980(a2, v6);
    v8 = (unsigned int)v24;
    if ( (unsigned int)v24 >= HIDWORD(v24) )
    {
      v19 = v7;
      sub_16CD150(&v23, v25, 0, 8);
      v8 = (unsigned int)v24;
      v7 = v19;
    }
    *(_QWORD *)&v23[8 * v8] = v7;
    v9 = (unsigned int)v27;
    LODWORD(v24) = v24 + 1;
    v10 = *(_QWORD *)v7;
    if ( (unsigned int)v27 >= HIDWORD(v27) )
    {
      v18 = v10;
      sub_16CD150(&i, v28, 0, 8);
      v9 = (unsigned int)v27;
      v10 = v18;
    }
    *(_QWORD *)&i[8 * v9] = v10;
    v6 = *(_QWORD *)(v6 + 16);
  }
  v11 = sub_126A190(*(_QWORD **)(a2 + 32), a4, 0, 0);
  v12 = (__int64 *)(a2 + 48);
  v13 = *(_QWORD *)(v11 + 24);
  if ( *(_BYTE *)(**(_QWORD **)(v13 + 16) + 8LL) )
  {
    v22 = 257;
    v15 = sub_1285290(v12, v13, v11, (int)v23, (unsigned int)v24, (__int64)v21, 0);
  }
  else
  {
    v22 = 257;
    sub_1285290(v12, v13, v11, (int)v23, (unsigned int)v24, (__int64)v21, 0);
    v14 = sub_1643350(*(_QWORD *)(a2 + 40));
    v15 = sub_15A06D0(v14);
  }
  v16 = i;
  *(_QWORD *)a1 = v15;
  *(_BYTE *)(a1 + 12) &= ~1u;
  *(_DWORD *)(a1 + 8) = 0;
  *(_DWORD *)(a1 + 16) = 0;
  if ( v16 != v28 )
    _libc_free(v16, v13);
  if ( v23 != v25 )
    _libc_free(v23, v13);
  return a1;
}
